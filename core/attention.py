import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor):
      """
      Q:[...,n,d_k]
      K:[...,m,d_k]
      V:[...,m,d_k]
      mask:[n,m]布尔矩阵，true为保留，false为隐蔽
      """
      d_k = Q.size(-1)

      #1.计算相似度分数
      #形状[...,n,m]
      scores=torch.einsum("...nk,...mk->...nm",Q,K)/math.sqrt(d_k)#张量除以常数时，pytorch会根据广播机制让每一个张量的元素都除以常熟

      #2.应用因果掩码，下三角矩阵，防止看到未来信息
      if mask is not None:
          scores=scores.masked_fill(mask==False,float('-inf'))

      #3.计算注意力权重，归一化
      #dim=-1对应的是每一个query对所有keyz的分布
      probs=torch.softmax(scores,dim=-1)

      #4.加权求和得到输出
      output=torch.einsum("...nm,...mk->...nk",probs,V)
      return output