import torch

## 1.basic 广播机制
a = torch.rand(3,4)
b = torch.rand(4)
print(a)
print(b)
print(a+b)

## 2.matmul乘法运算
a = torch.full((2,2),fill_value=3.)
b = torch.ones(2,2)
print(a,b)
print(a*b) #这样仅仅表示逐个元素对应位置相乘
print(a@b) # @就表示乘法
print(torch.matmul(a,b)) #通用的tensor乘法
c = torch.mm(a,b) # only for 2d也就是只适用于二维张量
print(c)

# 3.pow次方运算
# sqrt()求开方运算
# rsqrt()对每个元素取平方根再取倒数
# exp（） 指数运算
# log() 求对数运算
# floor()向下取整 ceil() 向上取整
#  round() 就近取整运算
# trunc() 标量x的截断值是最接近其的整数
# frac()  标量x的截断值是小数部分的值

# 4.clamp()
#clamp（）函数的功能将输入input张量每个元素的值压缩到区间[min, max]，并返回结果到一个新张量。
a = torch.rand(3,4)*15
print(a)
a = torch.clamp(a,0.,10.)
print(a)