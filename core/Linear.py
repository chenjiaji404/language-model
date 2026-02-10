import torch
import torch.nn as nn
from torch.nn import factory_kwargs


class Linear(nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super(Linear,self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        #1.准备工厂参数
        factory_kwargs={'device':device,'dtype':dtype}

        #2.占坑 定义权重参数W（形状：out*in）
        #注意：不用bias,更符合现代LLM做法
        self.weight = nn.Parameter(torch.empty(out_features,in_features,**factory_kwargs))

        #3.填充 执行截断正态分布初始化
        std=(2.0/(in_features+out_features))**0.5
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3*std, b=3*std)

    def forward(self,x:torch.Tensor):
        #x的形状中最后一维是in_feature
        # 使用einsum表达：输入的最后一位i和权重的最后一位i进行相乘，输出o
        # 这种写法比x@self.weight.T更有可读性，且支持任意batch
        return torch.einsum('...i,oi->...o',x,self.weight)#相消，i表示infeatures，oi前面的o是权重参数W的out_features，i是infratures
        #相消后得到最后一维是outfeatures，前面不变

