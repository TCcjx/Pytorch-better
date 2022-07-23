import torch


# 1.torch.norm 是对输入的Tensor求范数
# torch.norm(input,p=2),需要把要求的范数的tensor作为参数传入进去，默认求2范数
a = torch.ones((2,3))  #建立tensor
a2 = torch.norm(a)      #默认求2范数：所有元素平方和开根号
a1 = torch.norm(a,p=1)  #指定求1范数：所有元素的绝对值之和
print(a)
print(a2)
print(a1)

# 2.tensor直接调用norm函数：a.norm(p,dim=a)指定维度求p范数
a = torch.full([8], 1.)
b = a.view([2, 4])
c = a.view([2, 2, 2])

# 求L1范数(所有元素绝对值求和)
print(a.norm(1), b.norm(1), c.norm(1))
# 求L2范数(所有元素的平方和再开根号)
print(a.norm(2), b.norm(2), c.norm(2))

# 在b的1号维度上求L1范数，1维度指的是行数据
print(b.norm(1, dim=1))#输出tensor([4., 4.])
# 在b的1号维度上求L2范数
print(b.norm(2, dim=1))#输出tensor([2., 2.])

# 在c的0号维度上求L1范数
#dim=0是对0维度上的一个向量求范数，返回结果数量等于其列的个数
print(c.norm(1, dim=0))
# 在c的0号维度上求L2范数
print(c.norm(2, dim=0))

# 3.求张量中的mean，sum，min，max，prod
b = torch.arange(8).reshape(2,4).float()
print(b)
# 求均值，累加，最小，最大，累积
print(b.mean(), b.sum(), b.min(), b.max(), b.prod())
# 打平后的最小最大值索引
print(b.argmax(), b.argmin())

# 4.求前K大/前K小/第k小
print("-------")
d = torch.randn(2,10)
print(d.topk(3, dim=1))  # 最大的3个类别
print(d.topk(3, dim=1, largest=False))  # 最小的3个类别
print(d.kthvalue(8,dim=1))  # 求第8小(一共10个那就是第3大)

# 5.比较
# torch.eq()