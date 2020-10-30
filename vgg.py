import time
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import sys
import torchprof
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_printoptions(profile="full") 

class fashion_mnist():
    def get_fashion_mnist_labels(labels):
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                       'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]
    
    def load_data(batch_size, resize=None, root='./'):
        """Download the fashion mnist dataset and then load into memory."""
        trans = []
        if resize:
            trans.append(torchvision.transforms.Resize(size=resize))
        trans.append(torchvision.transforms.ToTensor())
        transform = torchvision.transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
        mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)
        train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
        test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_iter, test_iter

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
    
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return nn.Sequential(*blk)

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                 nn.Linear(fc_features, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, fc_hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(fc_hidden_units, 10)
                                ))
    return net

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
        
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    # 保存网络和网络训练参数
    torch.save(net, 'vgg.pkl')                     # 保存整个神经网络到vgg.pkl中
    torch.save(net.state_dict(), 'vgg_paras.pkl')  # 保存网络里的参数到vgg_paras.pkl中

# 载入模型并测试
def test():
    # 加载保存的模型和参数
    net = torch.load('vgg.pkl',map_location = 'cpu')

    # 网络移到GPU上
    net.eval()
    net = net.to(device)

    # 数据移动到GPU上
    cnt = 0
    gpu_input_data = []
    for i,j in test_iter:
        gpu_input_data.append(i.to(device))
        cnt += 1
        if cnt >= 20:
            break
    time.sleep(0.1)

    #进行推理验证
    cnt = 0
    for i in gpu_input_data:
        y = net(i)
        cnt += 1
        time.sleep(0.02)
        if cnt >= 20:
            break

ratio = 8
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), 
                    (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)

batch_size = 64
lr, num_epochs = 0.001, 30

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train_iter, test_iter = fashion_mnist.load_data(batch_size, resize=224)

with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=True) as prof:
    test()
prof.export_chrome_trace('./vgg.json')
exit(0)