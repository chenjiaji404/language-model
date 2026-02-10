import json
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from tqdm import tqdm
from TransformerLM import TransformerLM


# --- 1. SFT 数据集处理类 ---
class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, max_length=512):
        # 兼容你根目录下的分词器文件
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.max_length = max_length
        self.samples = []

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|padding|>"

        self.user_tokens = self.tokenizer.encode("User: ", add_special_tokens=False)
        self.assistant_tokens = self.tokenizer.encode("Assistant: ", add_special_tokens=False)
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 兼容不同格式：支持 conversations 列表或直接的 prompt/output
                if "conversations" in data:
                    self.samples.append(data["conversations"])
                elif "instruction" in data:  # 兼容你刚加入的身份数据格式
                    conv = [
                        {"role": "user", "content": data["instruction"]},
                        {"role": "assistant", "content": data["output"]}
                    ]
                    self.samples.append(conv)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        input_ids = []
        labels = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                encoded = self.tokenizer.encode(content, add_special_tokens=False)
                full_content = self.user_tokens + encoded
                input_ids.extend(full_content)
                labels.extend([-100] * len(full_content))
            elif role == "assistant":
                encoded = self.tokenizer.encode(content, add_special_tokens=False)
                full_content = self.assistant_tokens + encoded + [self.eos_token_id]
                input_ids.extend(full_content)
                labels.extend(full_content)

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            labels += [-100] * padding_len

        return torch.tensor(input_ids), torch.tensor(labels)


# --- 2. 推理验证函数（用于训练中监控乱码） ---
def evaluate_sync(model, tokenizer, device):
    model.eval()
    prompt = "User: 你是谁？ Assistant: "
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)]).to(device)

    with torch.no_grad():
        # 简单生成，看是否乱码
        out = model.generate(input_ids, max_new_tokens=32)

        res = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n[实时采样验证]: {res}")
    model.train()


# --- 3. 训练主函数 ---
def train_sft():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter("runs/minimind_sft")

    model_config = {
        "vocab_size": 6400,
        "max_seq_len": 512,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 1408,
        "rope_theta": 10000.0
    }

    model = TransformerLM(**model_config).to(device)

    # 路径改为 "./" 因为你的分词器就在根目录
    tok_path = "./"
    data_path = "./dataset-minimind/sft_mini_512.jsonl"

    train_dataset = SFTDataset(data_path, tok_path, max_length=model_config["max_seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


    pretrained_ckpt = "minimind_sft_final.pth"
    if os.path.exists(pretrained_ckpt):
        print(f"加载权重: {pretrained_ckpt}")
        model.load_state_dict(torch.load(pretrained_ckpt, map_location=device), strict=False)
    else:
        print("警告：未找到预训练权重，模型将从随机状态开始。")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    epochs = 2  # 增加到 2 轮，强化身份记忆
    global_step = 0


    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, (input_ids, labels) in enumerate(pbar):
            input_ids, labels = input_ids.to(device), labels.to(device)

            x = input_ids[:, :-1]
            y = labels[:, 1:]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                # 计算准确率
                preds = torch.argmax(logits, dim=-1)
                mask = (y != -100)
                accuracy = (preds == y)[mask].float().mean() if mask.any() else torch.tensor(0.0)

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止乱码
            scaler.step(optimizer)
            scaler.update()

            if step % 10 == 0:
                writer.add_scalar("Loss/SFT", loss.item(), global_step)
                writer.add_scalar("Accuracy/SFT", accuracy.item(), global_step)
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{accuracy.item():.4f}"})

            # 每 100 步打印一次采样，监控是否认得自己
            if step % 100 == 0:
                # 注意：如果你的 TransformerLM 没有实现 generate 方法，请注释掉下面这行
                try:
                    evaluate_sync(model, train_dataset.tokenizer, device)
                except Exception as e:
                    pass

            global_step += 1

        torch.save(model.state_dict(), f"checkpoint_sft_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "minimind_sft_final.pth")
    writer.close()
    print("SFT 训练完成。")


if __name__ == "__main__":
    train_sft()