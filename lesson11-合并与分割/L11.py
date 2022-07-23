import torch
## 1.cat()合并某个维度的值

a = torch.rand(4,32,8)
b = torch.rand(5,32,8)
c = torch.cat((a,b),dim=0)
print(c.shape)

## 2.stack()方法,创建新的维度

a1 = torch.rand(4,3,16,32)
a2 = torch.rand(4,3,16,32)
a3 = torch.stack([a1,a2],dim=2)
print(a3.shape)

## 3.split()

a = torch.rand(2,32,8)
aa,bb = a.split((1,1),dim=0)
print(aa.shape,bb.shape)

## 4.chunk()
a = torch.rand(2,32,8)
aa,bb = a.chunk(2,dim=2)
print(aa.shape,bb.shape)