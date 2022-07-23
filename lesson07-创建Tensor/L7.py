import numpy as np
import torch

# 1.从np数组创建tensor
a = np.array([2,3,3])
print(a,type(a))
b = torch.from_numpy(a)
print(b,b.type())

# 2.由list创建
a = torch.tensor([2,3.2])
print(a,a.shape)
b = torch.FloatTensor(2,3)
print(b)

# 3.由rand创建
a = torch.rand(3,3) #[0,1]之间
print(a)
b = torch.rand_like(a)
c = torch.randint(2,5,(3,3)) #[min,max)
print(b,c)

# 4.randn创建
# N(0,1) N(u,std) 均值为0，标准差为1
a = torch.randn(3,3)
print(a)