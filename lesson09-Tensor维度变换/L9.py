import torch

# 1.通过view改变尺寸
a = torch.rand(4,1,28,28)
print(a.shape)
a = a.view(4,28*28)
print(a.shape)

# 2.unsqueeze扩展维度
print(a.shape)
a = a.unsqueeze(0)
print(a.shape)

# 3.sequeeze压缩维度
print(a.shape) # [1,4,784]
a = a.squeeze(0)
print(a.shape) # [4,784]

# 4.expand在指定某个维度上进行复制扩充
a = torch.rand(1,28,1,1)
print(a.shape)
a = a.expand(4,28,4,4) # 只可以从 1-> n；不可以从 其他 - > n ,如果设置成-1,则表示原有维度尺寸不变
print(a.shape)

## 5.repeat
b = torch.rand(4,1,22,2)
b = b.repeat(1,1,1,2)
print(b.shape) #比较占用内存空间，不推荐使用

## 6..t()方法
a = torch.rand(3,4)
print(a.t()) #获取转置矩阵

## 7.Transpose()
"""
contiguous：view只能用在contiguous的variable上。
如果在view之前用了transpose, permute等，
需要用contiguous()来返回一个contiguous copy。
"""
a1 = torch.rand(4,3,32,32)
a2 = a1.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)
print(torch.all(torch.eq(a2, a1))) #判断两个tensor是否相同


## 8.permute()
x = torch.rand(4,3,28,28) # b,c,h,w
x1 = x.permute(0,2,3,1) # b,h,w,c
print(x1.shape)

