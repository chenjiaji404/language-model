import torch
import torch.nn as nn
from torch.nn import factory_kwargs


class LayerNorm(nn.Module):#对一个完整向量所有维度求均值方差归一化
    def __init__(self, d_model, eps:float=1e-8,device=None,dtype=None):
        super(LayerNorm,self).__init__()
        factory_kwargs={"device":device,"dtype":dtype}

        #1.学习参数初始化
        #weight(gamma)：缩放参数，初始化全为1，可学习
        self.weight = nn.Parameter(torch.ones(d_model,**factory_kwargs))#长度为d_model的一维向量
        #bias(beta)：偏移参数，初始化全为0 Layernorm独有的
        self.bias = nn.Parameter(torch.zeros(d_model,**factory_kwargs))
        self.eps = eps

    def forward(self, x:torch.Tensor):
        #x形状：[batah_size，seq_len,d_model]
        in_dtype=x.dtype
        #转换为float32确保计算均值和方差的稳定性
        x_float=x.to(torch.float32)

        #计算均值 对最后一个维度（特征维）求平均，keep_dim=True用于后续减法广播
        mean=x_float.mean(dim=-1,keepdim=True)

        #计算方差 注意：这里使用biased variance,与pytorch官方nn.Layernorm对齐
        var=x_float.var(dim=-1,keepdim=True,unbiased=False)

        #归一化
        x_normed=(x_float-mean)/torch.sqrt(var+self.eps)#self.eps很小，避免除0

        #应用可学习的weight和bias
        result=x_normed*self.weight + self.bias
        return result.to(in_dtype)


class RMSNorm(nn.Module):#计算更少，更稳定，只需计算均方根 现代大模型标配
    def __init__(self, d_model, eps:float=1e-8,device=None,dtype=None):
        super(RMSNorm,self).__init__()
        factory_kwargs={"device":device,"dtype":dtype}
        #必须初始化全为1
        self.weight = nn.Parameter(torch.ones(d_model,**factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        # 转换为float32防止溢出
        x_float = x.to(torch.float32)

        #计算均方根
        #keep_dim=True 方便后续除法自动广播 保留计算维度形状，不用手动调整形状
        ms=x_float.pow(2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(ms+self.eps)

        #归一化并乘以可学习参数g
        result=(x_float/rms)*self.weight

        return result.to(in_dtype)

