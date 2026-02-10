import torch
import torch.nn as nn
from Linear import *
from attention import *
from RotaryPositionalEmbedding import *
from MultiHeadAttention import MultiHeadAttention
from SwiGLU import *
from LayerNormRMSNorm import RMSNorm,LayerNorm
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff:int,max_seq_len,theta:float,device=None,dtype=None,#新增实验参数
                 use_rms_norm:bool=True,
                 norm_mode:str="pre",
                 ffn_type:str="swiglu"):
        super(TransformerBlock,self).__init__()

        self.norm_mode = norm_mode.lower()

        self.attn=MultiHeadAttention(d_model=d_model,num_heads=num_heads,max_seq_len=max_seq_len,theta=theta,device=device,dtype=dtype)
        #初始化两个RMSNorm层，分别用于attention和ffn
        #现在改成参数选择
        norm_cls = RMSNorm if use_rms_norm else LayerNorm
        self.ln1=norm_cls(d_model=d_model,device=device,dtype=dtype)
        self.ln2=norm_cls(d_model=d_model,device=device,dtype=dtype)

        #扩展对不同FFN支持
        if ffn_type.lower() == "swiglu":
            self.ffn=SwiGLU(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)
        else:
            self.ffn = nn.Sequential(
                Linear(d_model, d_ff, device=device, dtype=dtype),
                nn.GELU(),
                Linear(d_ff, d_model, device=device, dtype=dtype)
            )

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        #pre和post
        if self.norm_mode == "pre":
            # Pre-Norm 结构: x + Sublayer(Norm(x))
            #attention子层
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            #ffn子层
            x = x + self.ffn(self.ln2(x))
        elif self.norm_mode == "post":
            # Post-Norm 结构: Norm(x + Sublayer(x))
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

        return x
