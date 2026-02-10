import torch
import torch.nn as nn
from torch.nn import factory_kwargs


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,device=None,dtype=None):
        super(Embedding, self).__init__()
        #1.分配内存并包装为参数
        factory_kwargs={'device':device,'dtype':dtype}
        self.weight=nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))

        #2.初始化 mean=0.0 std=1.0 截断在【-3，3】
        nn.init.trunc_normal_(self.weight,mean=0.0,std=0.01,a=-3.0,b=3.0)

    def forward(self,token_ids:torch.Tensor):
        return self.weight[token_ids]#索引，对token_ids的每一个数字找对应的向量，组合 返回形状【B,S,D】

