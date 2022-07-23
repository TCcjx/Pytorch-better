import torch

# 1.索引访问
a = torch.rand(5,3,28,28)
print(a[0].shape)
print(a[0,0].shape)

# 2.切片访问
print(a[:2].shape)
print(a[::2].shape) # start:end:step

# 3.select by mask
print(a)
mask = a.ge(0.5) #大于的值为1，小于的值为0
print(mask)

