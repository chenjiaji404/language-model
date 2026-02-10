import torch
import uvicorn
import os
import json
import faiss
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from TransformerLM import TransformerLM

app = FastAPI(title="Xiao Chen AI Service with RAG")


# --- 1. RAG 管理类 ---
class RAGManager:
    def __init__(self, data_path, index_path="rag_index.faiss", model_name='shibing624/text2vec-base-chinese'):
        self.data_path = data_path
        self.index_path = index_path
        # 使用专门针对中文优化的轻量级 Embedding 模型
        print(f"正在加载 Embedding 模型: {model_name}...")
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.sentences = []

        self._prepare_data()

    def _prepare_data(self):
        # 1. 解析数据：将三元组转化为自然语言句子
        if not os.path.exists(self.data_path):
            print(f"错误：找不到数据文件 {self.data_path}")
            return

        print("正在预处理知识库数据...")
        raw_sentences = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    parts = [p.strip() for p in item['answer'].split('|||')]
                    if len(parts) == 3:
                        # 转换成自然语言：例如 "守望星光的出品公司是韩玉玲原创音乐。"
                        s = f"{parts[0]}的{parts[1]}是{parts[2]}。"
                        raw_sentences.append(s)
                except:
                    continue

        # 去重，减小索引压力
        self.sentences = list(set(raw_sentences))

        # 2. 构建或加载索引
        if os.path.exists(self.index_path):
            print("加载本地已有的向量索引...")
            self.index = faiss.read_index(self.index_path)
        else:
            print(f"正在构建向量索引（共 {len(self.sentences)} 条知识）...")
            embeddings = self.embed_model.encode(self.sentences, show_progress_bar=True)
            dimension = embeddings.shape[1]# 维度，通常是 768
            self.index = faiss.IndexFlatL2(dimension)# 创建一个基于 L2 距离（欧氏距离）的索引   索引 = 坐标值（数据）+ 快速查找算法（结构）。
            self.index.add(np.array(embeddings).astype('float32'))# 把所有坐标存进去
            faiss.write_index(self.index, self.index_path)
            print("索引构建完成并保存。")

    def search(self, query, top_k=3):
        """检索最相关的背景知识"""
        if self.index is None:
            return ""
        query_vec = self.embed_model.encode([query])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), top_k)

        context_list = [self.sentences[i] for i in indices[0] if i != -1]#indices（索引）：这是最关键的，它返回了最接近的那几个知识点在原始列表里的“行号”。
        #动作：刚才 FAISS 只给了我们行号（比如 [5, 12, 88]），而模型看不懂数字。这一行代码根据行号，回到了你最初存句子的那个列表（self.sentences）里，把那 3 句真正的中文字符串给取了出来
        return "\n".join(context_list)


# --- 2. 模型加载配置 (保持你原来的部分) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "./"
MODEL_PATH = "sft_pre_rmsnorm_weight/minimind_sft_final.pth"

model_config = {
    "vocab_size": 6400,
    "max_seq_len": 512,
    "d_model": 512,
    "num_layers": 8,
    "num_heads": 8,
    "d_ff": 1408,
    "rope_theta": 10000.0,
    "use_rms_norm": True,
    "norm_mode": "pre"
}

print(f"正在加载小陈模型至 {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = TransformerLM(**model_config).to(DEVICE)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("模型加载成功！")
else:
    print(f"警告：未找到权重文件 {MODEL_PATH}")

# 初始化 RAG 管理器
# 第一次运行会比较慢（需要计算向量并保存），之后启动会非常快
model_local_path = r"D:\大模型\RAG向量化模型"

rag = RAGManager(data_path="RAG.json", model_name=model_local_path)


# --- 3. 路由定义 ---

@app.get("/", response_class=HTMLResponse)
async def get_portal():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>未找到 index.html，请确保它与脚本在同一目录。</h3>"


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    temp = data.get("temperature", 0.7)

    # --- RAG 核心逻辑 ---
    # 1. 检索背景知识
    # 强制将检索数量降为 1，小模型处理不了多条信息的冲突
    context = rag.search(prompt, top_k=1)
    context ="北京是中国的首都。"
    # 2. 缩减指令，不要用“请根据背景回答”这种长指令，直接用符号对齐
    if context:
        # 使用最少的字符引导模型
        # 别加多余的换行和废话
        full_prompt = f"User:参考:{context}问:{prompt} Assistant:"
    else:
        full_prompt = f"问题:{prompt}\n回答:"

    # --- 推理逻辑 ---
    input_ids = torch.tensor([tokenizer.encode(full_prompt, add_special_tokens=False)]).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            temperature=temp,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "response": response.strip(),
        "context": context  # 返回检索到的知识，方便你调试
    }


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)