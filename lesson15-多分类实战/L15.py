import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.nn as nn
from torch.utils.data import DataLoader

w1 ,b1 = torch.randn(200,784,requires_grad=True),\
         torch.zeros(200,requires_grad=True)
w2,b2 = torch.randn(200,200,requires_grad=True),\
        torch.zeros(200,requires_grad=True)
w3,b3 = torch.randn(10,200,requires_grad=True),\
        torch.zeros(10,requires_grad=True)

train = datasets.MNIST(
    r"E:\python_project\new_study\Pytorch-better\lesson15-多分类实战",train=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),download= True
)

test = datasets.MNIST(
    r"E:\python_project\new_study\Pytorch-better\lesson15-多分类实战",train=False,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),download= True
)

train_loader = DataLoader(train,batch_size=256,shuffle=True)
test_loader = DataLoader(test,batch_size=256,shuffle=True)

def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x

opt = optim.SGD([w1,b1,w2,b3,w3,b3],lr=0.01)
loss_fn = nn.CrossEntropyLoss()

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)
epochs = 20
for t in range(epochs):
    train_loss = 0
    train_correct = 0
    train_label = 0
    for x,y in train_loader:
        x = x.view(-1,28*28)
        y_pred = forward(x)
        loss = loss_fn(y_pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item()
        y_pred = torch.argmax(y_pred,dim=1)
        train_correct += (y_pred==y).sum().item()
        train_label += y.size(0)
    print(f'epoch {t+1}, loss : {train_loss} ,acc : {train_correct/train_label}')
    test_loss = 0
    test_correct = 0
    test_label = 0
    for x,y in test_loader:
        x = x.view(-1,28*28)
        y_pred = forward(x)
        test_loss += loss_fn(y_pred,y)
        y_pred = torch.argmax(y_pred,dim=1)
        test_correct += (y_pred==y).sum().item()
        test_label += y.size(0)
    print(f'epoch {t + 1}, loss : {test_loss} ,acc : {test_correct / test_label}')
