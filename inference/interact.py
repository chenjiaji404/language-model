import torch
import uvicorn
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer
from TransformerLM import TransformerLM

app = FastAPI(title="Xiao Chen AI Service")

# --- 1. 跨域配置 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. 模型加载配置 ---
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

# --- 3. 路由定义 ---

@app.get("/", response_class=HTMLResponse)
async def get_portal():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>未找到 index.html，请确保它与脚本在同一目录。</h3>"

@app.post("/chat")
async def chat(request: Request):#request是fastapi封装好的一个对象，包含了前端浏览器发送给后端服务器的所有信息
    data = await request.json()#前端发送给后端的所有数据，前端会把文字等打包成json格式，再发给后端
    prompt = data.get("prompt", "")#用户输入的问题，await告诉程序我要执行这个操作了，你先去忙别的，等操作完了再叫我
    temp = data.get("temperature", 0.7)


    identity_keywords = ["你是谁", "你叫什么", "你的名字", "谁训练了你", "谁创造了你", "你的作者"]
    if any(kw in prompt for kw in identity_keywords):
        return {
            "response": "我是小陈，是由陈嘉稷训练的AI助手。"
        }
    # -----------------------

    # 正常的推理逻辑
    full_prompt = f"User: {prompt} Assistant: "
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
        "response": response.strip()
    }#通过/chat返回到前端，也是处理成json形式

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)