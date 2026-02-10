import torch
import torch.nn as nn
import torch.nn.functional as F
from Linear import *
def silu_fn(in_features):
    return in_features*torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff:int,device=None,dtype=None):
        super(SwiGLU,self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        #W1和W3是并行生维层：d_model->d_ff
        self.w1=Linear(d_model,d_ff,device,dtype)
        self.w3=Linear(d_model,d_ff,device,dtype)
        #W2是降维层
        self.w2=Linear(d_ff,d_model,device,dtype)

    def forward(self,x:torch.Tensor):
       gate=silu_fn(self.w1(x))
       signal=self.w3(x)

       return self.w2(signal*gate)
