import os
import json
import torch
import torch.nn as nn
import time
import re  # 新增：用于正则匹配文件名序号
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from tqdm import tqdm
from TransformerLM import TransformerLM


# [PretrainDataset 类保持不变，逐字保留]
class PretrainDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.max_length = max_length
        self.samples = []

        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|padding|>"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集未找到: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    text += self.tokenizer.eos_token if self.tokenizer.eos_token else ""
                    self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return encodings['input_ids'].squeeze(0)


def train():
    # --- 1. 环境设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter(log_dir="./runs/minimind_pretrain")

    # --- 2. 准备数据 ---
    data_path = os.path.join("dataset-minimind", "pretrain_hq.jsonl")
    dataset = PretrainDataset(
        file_path=data_path,
        tokenizer_path="./",
        max_length=512
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Tokenizer 词表大小: {len(dataset.tokenizer)}")

    # --- 3. 模型初始化 ---
    model = TransformerLM(
        vocab_size=6400,
        max_seq_len=512,
        d_model=512,
        num_layers=8,
        num_heads=8,
        d_ff=1408,
        rope_theta=10000.0,
        device=device,
        use_rms_norm=True,
        norm_mode="pre"
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    # --- 4. 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    pad_id = dataset.tokenizer.pad_token_id if dataset.tokenizer.pad_token_id is not None else 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # --- 5. 自动寻找最新的 checkpoint 并加载 ---
    start_epoch = 0
    global_step = 0
    saved_checkpoints = []

    # 扫描当前目录下所有 checkpoint_step_X.pth
    ckpt_files = [f for f in os.listdir('.') if f.startswith('checkpoint_step_') and f.endswith('.pth')]
    if ckpt_files:
        # 按文件名中的数字序号排序
        ckpt_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        latest_ckpt = ckpt_files[-1]

        print(f"检测到历史权重: {latest_ckpt}，正在恢复训练...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        # 计算粗略的开始 epoch (根据 step 和 loader 长度)
        start_epoch = global_step // len(train_loader)

        # 将现有的 checkpoints 加入管理列表，确保后续清理逻辑正常
        saved_checkpoints = ckpt_files[-5:]
    else:
        print("未发现历史权重，从头开始训练。")

    # --- 6. 训练循环 ---
    model.train()
    print(f"开始预训练，起始 Step: {global_step}，日志保存至 ./runs/")

    for epoch in range(start_epoch, 3):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        start_time = time.time()

        for step, batch in enumerate(pbar):
            # 如果是恢复训练，跳过当前 epoch 中已经跑过的 step
            if epoch == start_epoch and step < (global_step % len(train_loader)):
                continue

            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                preds = torch.argmax(logits, dim=-1)
                mask = (y != pad_id)
                accuracy = (preds == y)[mask].float().mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if step % 10 == 0:
                end_time = time.time()
                tokens_processed = 10 * batch.size(0) * batch.size(1)
                tps = tokens_processed / (end_time - start_time)
                start_time = time.time()

                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Accuracy/train", accuracy.item(), global_step)
                writer.add_scalar("Tokens_per_sec", tps, global_step)

                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{accuracy.item():.4f}",
                    "TPS": f"{int(tps)}"
                })

            # 保存与清理逻辑
            if global_step > 0 and global_step % 500 == 0:
                ckpt_path = f"checkpoint_step_{global_step}.pth"
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, ckpt_path)

                saved_checkpoints.append(ckpt_path)
                if len(saved_checkpoints) > 5:
                    old_ckpt = saved_checkpoints.pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            global_step += 1

    torch.save(model.state_dict(), "minimind_pretrain_final.pth")
    writer.close()


if __name__ == "__main__":
    train()