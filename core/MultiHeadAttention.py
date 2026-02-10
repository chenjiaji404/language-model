import torch
import torch.nn as nn
from einops import rearrange
from Linear import Linear
from attention import scaled_dot_product_attention
from RotaryPositionalEmbedding import *
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads,max_seq_len,theta,device,dtype,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        #维度校验
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        #1.Q,K,V投影层，将输入映射到三个不同的特征空间
        self.q_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj=Linear(d_model,d_model,device=device,dtype=dtype)

        #2.输出投影层，整合所有信息
        self.out_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.dropout=nn.Dropout(p=dropout)

        #3.Rope初始化：仅在提供theta时使用
        if theta is not None and max_seq_len is not None:
            self.rope=RotaryPositionalEmbedding(theta,self.d_k,max_seq_len,device=device)
        else:
            self.rope=None

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor=None):
            b,s,d=x.shape

            #1&2:线性投影并拆分多头
            #使用einops+rearrange代替view+transpose
            #语义：将长度为d的特征拆分成（h d_k)，并将h维移动到序列s维之前
            q=rearrange(self.q_proj(x),"... s (h d) -> ... h s d",h=self.num_heads)
            k=rearrange(self.k_proj(x),"... s (h d) -> ... h s d",h=self.num_heads)
            v=rearrange(self.v_proj(x),"... s (h d) -> ... h s d",h=self.num_heads)

            #3应用ROPE旋转位置编码
            if self.rope is not None:
                if token_positions is None:
                   #默认生成从o开始的维度
                   #expand处理batah维度，不额外占用物理内存
                   #对q,k进行旋转，v保持不变
                   q=self.rope(q,token_positions)
                   k=self.rope(k,token_positions)

            #4.生成因果掩码 下三角矩阵
            #确保q只能看到当前和以前的key的信息
            mask=torch.tril(torch.ones(s,s,device=x.device,dtype=torch.bool))

            #5注意力机制计算
            attn_out=scaled_dot_product_attention(q,k,v,mask=mask)

            #6合并多头
            attn_out=rearrange(attn_out,"... h s d -> ... s (h d)",h=self.num_heads)

            return self.out_proj(attn_out)