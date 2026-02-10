import torch
import torch.nn as nn
from Linear import *
from attention import *
from RotaryPositionalEmbedding import *
from MultiHeadAttention import MultiHeadAttention
from SwiGLU import *
from LayerNormRMSNorm import RMSNorm
from Embedding import *
from TransformerBlock import *
class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int,max_seq_len:int,d_model,num_layers,num_heads,d_ff:int,rope_theta:float,device=None,dtype=None,
                 #新增实验参数
                 use_rms_norm:bool=True,
                 norm_mode:str="pre",
                 ffn_type:str="swiglu"
                 ):
        super(TransformerLM,self).__init__()
        self.max_seq_len = max_seq_len
        #1.Embedding层
        self.token_embeddings = Embedding(vocab_size,d_model,device=device,dtype=dtype)

        #2.Transformer块
        self.layers=nn.ModuleList([TransformerBlock(
            d_model,num_heads,d_ff,max_seq_len,rope_theta,device,dtype,use_rms_norm=use_rms_norm,
            norm_mode=norm_mode,ffn_type=ffn_type
        )
        for _ in range(num_layers)])

        #3.最终的输出层
        if use_rms_norm:
            self.ln_final=RMSNorm(d_model=d_model,device=device,dtype=dtype)
        else:
            '''
            forward(input):
              return input
            '''
            self.ln_final=nn.Identity()
        self.lm_head=nn.Linear(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self,token_ids:torch.Tensor):
        b,s=token_ids.shape

        #准备信息用于RoPE [s]->[1,s]->[b,s]
        token_positions=torch.arange(s,device=token_ids.device).unsqueeze(0).expand(b,s)

        #1.Embedding
        x=self.token_embeddings(token_ids)

        #2.逐层通过TransformerBlock
        for layer in self.layers:
            x=layer(x,token_positions=token_positions)

        #3.最终归一化 如果use_rms_norm是false，这里就是直通
        x=self.ln_final(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(self,
                 token_ids: torch.Tensor,
                 max_new_tokens: int,
                 do_sample: bool = True,
                 temperature: float = 1.0,
                 eos_token_id: int = None):
        """
        自回归生成函数
        """
        # 确保输入为 Long 类型
        curr_token_ids = token_ids.long()

        for _ in range(max_new_tokens):
            # 1. 物理长度检查
            if curr_token_ids.size(1) >= self.max_seq_len:
                break

            # 2. 前向传播：获取当前序列所有位置的 logits
            # 内部会自动根据 curr_token_ids 的长度生成对应的 token_positions
            logits = self.forward(curr_token_ids)

            # 3. 提取最后一个时间步的分布 [batch, seq_len, vocab_size] -> [batch, vocab_size] 只拿出当前序列最后一个位置的预测结果，也就是下一个字，实现续写效果
            next_token_logits = logits[:, -1, :]

            # 4. 采样/贪婪搜索策略
            if not do_sample:
                # 贪婪搜索
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)#next——token就是概率最高的索引
            else:
                # 采样搜索
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # 5. 拼接新 token 准备进行下一轮预测
            curr_token_ids = torch.cat([curr_token_ids, next_token], dim=-1)

            # 6. 停止符检查
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return curr_token_ids

