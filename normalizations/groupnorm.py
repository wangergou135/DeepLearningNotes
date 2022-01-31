import numpy as np
import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)

        return x

if __name__ == "__main__":

    x=torch.randn([2,10,3,3])+1
    # Torch集成的方法
    m1 = torch.nn.GroupNorm(num_channels=10,num_groups=2)  # is true as default
    m2 = GroupNorm(num_features=10,num_groups=2)  # is true as default

    # 先计算前面五个通道的均值
    firstDimenMean = torch.Tensor.mean(x[0,0:5])
    # 先计算前面五个通道的方差
    firstDimenVar= torch.Tensor.var(x[0,0:5], False)  
    # 减去均值乘方差
    y = ((x[0][0][0][1] - firstDimenMean)/(torch.pow(firstDimenVar + m1.eps,0.5) )) * m1.weight[0] + m1.bias[0]
    print(y)

    y1=m1(x)
    # print(m1.weight)
    # print(m1.bias)
    print(y1[0,0,0,1])

    y2=m2(x)
    # print(m2.weight)
    # print(m2.bias)
    print(y2[0,0,0,1])