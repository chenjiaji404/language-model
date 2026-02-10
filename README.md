小陈 (Xiao Chen) AI —— 从零构建的轻量级全栈大语言模型项目

由小陈独立开发并训练的一个全栈大语言模型项目，旨在帮助有兴趣的同学更好的了解大模型底层原理、训练流程及部署环节。

• 数据集来源：该项目的数据集主要源自于 MiniMind 开源项目。

• 架构灵感：模型框架的实现灵感来源于斯坦福 CS336 (Language Modeling from Scratch) 课程作业。

• 底层实现：本项目独立实现了所有的 Transformer 核心模块（包括 Attention、RMSNorm、RoPE 等），而非直接调用现有的库封装。

 

 全链路实现方案：

该项目实现了从底层架构到用户交互的完整技术栈，具体包括：

1\. 模型搭建 (Model Architecture)

• 独立模块实现：本项目独立实现了所有的 Transformer 核心模块，而非直接调用现有的库封装。

• 核心组件：包括基于 einsum 的线性层 (Linear)、嵌入层 (Embedding)、以及集成了 RoPE (旋转位置编码) 的多头注意力机制。

• 现代优化：采用了 RMSNorm 归一化和 SwiGLU 激活函数，提升训练的稳定性和模型表达能力。



2\. 模型训练 (Training Pipeline)

• 词法分析：自实现并训练了 BPE Tokenizer，支持特殊 Token 的扩展与词表导出。

• 完整流水线：涵盖了从大规模语料的预训练 (Pretrain)、指令对齐的微调 (SFT) 到追求人类偏好一致性的 DPO 对齐。



3\. 模型部署 (Inference \& Deployment)

• 后端接口：基于 FastAPI 搭建了高性能的推理后端，支持异步请求处理。

• RAG 增强：集成了基于 FAISS 向量数据库的检索增强生成（RAG）功能，使模型能基于外部知识库回答问题。



4\. 前端界面 (Frontend Interface)

• Web 对话系统：使用 TailwindCSS 构建了一个响应式的对话界面。

• 实时调节：界面支持对 Temperature 等生成参数的实时调节，并完整展示了小陈 AI 的身份标识与作者信息。



项目结构：

.

├── core/           # 核心模型组件

│   ├── attention.py

│   ├── Embedding.py

│   ├── LayerNormRMSNorm.py

│   ├── Linear.py

│   ├── MultiHeadAttention.py

│   ├── RotaryPositionalEmbedding.py

│   ├── SwiGLU.py

│   ├── TransformerBlock.py

│   └── TransformerLM.py

├── tokenizer/      # BPE 分词器

│   ├── bpe\_tokenizer.py

│   ├── tokenizer.py

│   └── tokenizer\_config.json

├── train/          # 训练与微调

│   ├── pretrain.py

│   ├── sft.py

│   └── DPO.py

├── inference/      # 推理与交互

│   ├── chat.py

│   ├── interact.py

│   └── interact\_with\_rag.py

├── rag/            # RAG 检索增强

│   ├── RAG.py

│   

├── README.md

├── 模型效果.png

└── requirements.txt



 技术规格：

1\. 模型配置 (Model Config)

基于项目中定义的默认参数：

• 词表大小 (Vocab Size): 6400

• 隐藏层维度 (d\_model): 512

• 层数 (Layers): 8

• 注意力头数 (Heads): 8

• 前馈网络维度 (d\_ff): 1408

• 最大序列长度: 512

注意事项：本项目代码文件的路径需要修改





 


