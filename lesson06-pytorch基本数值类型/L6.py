import torch
a = torch.randn(2,3)
b = torch.ones(2,4,dtype=torch.float64)
print(b,b.shape)
print(b.type())
print(a.type())