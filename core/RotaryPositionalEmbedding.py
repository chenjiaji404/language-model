import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):#theta是基准频率
        super(RotaryPositionalEmbedding,self).__init__()
        self.d_k=d_k

        #1.计算频率omega_k=theta^(-2k/d)
        #我们只需要计算d_k/2个频率，因为是成对出现的
        #arange(0,d_k,2)对应公式中的2k-2
        powers=torch.arange(0,d_k,2,device=device).float()/d_k
        freqs=1.0/(theta**powers)#形状（d_k/2,)

        #2.创建位置序列[0,1...max_seq_len-1]
        t=torch.arange(max_seq_len,device=device).float()#形状（max_seq_len,)

        #3.计算所有位置的角度（外积）
        freqs_metric=torch.outer(t,freqs)#形状（max_seq_len,d_k/2)

        #4.预计算cos sin并作为buffer注册
        #使用persistent=False确保这些缓存不会被保存在state_dict中(因为可以随时重新生成）
        self.register_buffer('cos_couched',freqs_metric.cos(),persistent=False)#绑定了属性
        self.register_buffer('sin_couched',freqs_metric.sin(),persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        #1.提取cos sin(...,seq,d_k/2)
        cos=self.cos_couched[token_positions]
        sin=self.sin_couched[token_positions]#输出为cos(m*角度）m为位置索引

        #2.维度对齐
        #只有当x是4D（含head维）且cos是3D（含batch维）时，才需要手动插入head维
        #对于test_rope这种3D x vs 2D cos的情况，pytorch会自动左侧补1，无需对齐操作
        if x.ndim>cos.ndim and cos.ndim>3:
            cos=cos.unsqueeze(1)
            sin=sin.unsqueeze(1)

        #确保类型一致
        cos=cos.to(x.dtype)
        sin=sin.to(x.dtype)

        #3.拆分
        x_even=x[...,0::2]
        x_odd=x[...,1::2]

        out_put=torch.empty_like(x)

        out_put[...,0::2]=x_even*cos-x_odd*sin
        out_put[...,1::2]=x_even*sin+x_odd*cos

        return out_put
