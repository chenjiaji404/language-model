import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F


from TransformerLM import TransformerLM


# ==========================================
# 板块 1：DPO 偏好数据集处理
# ==========================================
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer_path, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True,local_files_only=True )
        self.max_length = max_length
        self.samples = []

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = "<|padding|>"

        # 按照 SFT 阶段的格式定义模板
        self.prompt_template = "User: {} Assistant: "

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 数据集格式：{"prompt": "...", "chosen": "...", "rejected": "..."}
                if all(k in data for k in ("prompt", "chosen", "rejected")):
                    self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def encode_pair(self, prompt, answer):
        """
        将 prompt 和 answer 拼接，并对 prompt 部分进行 Mask (-100)
        """
        prompt_text = self.prompt_template.format(prompt)
        full_text = prompt_text + answer + self.tokenizer.eos_token

        # 编码完整序列和仅 Prompt 序列
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # 截断
        input_ids = full_ids[:self.max_length]
        prompt_len = len(prompt_ids)

        # 构建 labels：Prompt 部分填充 -100，只有 Answer 部分保留 id
        labels = ([-100] * prompt_len + input_ids[prompt_len:])[:self.max_length]

        # Padding 处理
        padding_len = self.max_length - len(input_ids)
        if padding_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_len
            labels += [-100] * padding_len

        return torch.tensor(input_ids), torch.tensor(labels)

    def __getitem__(self, idx):
        item = self.samples[idx]
        # 分别处理“选中的”和“拒绝的”回答
        c_ids, c_labels = self.encode_pair(item['prompt'], item['chosen'])
        r_ids, r_labels = self.encode_pair(item['prompt'], item['rejected'])

        return {
            "chosen_ids": c_ids,
            "chosen_labels": c_labels,
            "rejected_ids": r_ids,
            "rejected_labels": r_labels
        }


# ==========================================
# 板块 2：DPO 核心算法组件
# ==========================================
def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor):
    """
    计算序列中有效 token 的累计对数概率 (Log Probabilities)
    """
    # 1. 预测平移：logits 的第 t 个输出对应 labels 的第 t 个 token
    # logits: [batch, seq, vocab] -> [batch, seq-1, vocab]
    # labels: [batch, seq] -> [batch, seq-1]
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    # 2. 准备掩码：只计算 labels 不为 -100 的位置
    loss_mask = (labels != -100)

    # 3. 提取对应 label 的 log_softmax 概率
    # 使用 gather 提取每个位置实际 token 对应的 logit
    per_token_logps = torch.gather(
        F.log_softmax(logits, dim=-1),
        dim=2,
        index=labels.clamp(min=0).unsqueeze(2)
    ).squeeze(2)

    # 4. 对序列维度求和，得到每个样本的总 logp
    return (per_token_logps * loss_mask).sum(-1)


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """
    DPO 损失计算：L_DPO = -E[log sigmoid(beta * ((log p_theta(y_w|x) - log p_ref(y_w|x)) - (log p_theta(y_l|x) - log p_ref(y_l|x))))]
    """
    # 计算策略模型与参考模型之间的对数比值
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios

    # DPO 损失
    loss = -F.logsigmoid(beta * logits).mean()

    # 监控指标：奖励准确率（即策略模型给 Chosen 的评分高于 Rejected 的频率）
    reward_acc = (pi_logratios > ref_logratios).float().mean()

    return loss, reward_acc


# ==========================================
# 板块 3：DPO 训练主循环
# ==========================================
def train_dpo():
    # --- 环境配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler('cuda')
    writer = SummaryWriter("runs/minimind_dpo")

    # DPO 超参数
    beta = 0.1
    learning_rate = 5e-7  # DPO 通常使用极小的学习率

    model_config = {
        "vocab_size": 6400,
        "max_seq_len": 512,
        "d_model": 512,
        "num_layers": 8,
        "num_heads": 8,
        "d_ff": 1408,
        "rope_theta": 10000.0
    }

    # --- 模型初始化 ---
    # 1. 初始化当前策略模型
    model = TransformerLM(**model_config).to(device)
    # 2. 初始化参考模型
    ref_model = TransformerLM(**model_config).to(device)

    # 加载 SFT 阶段训练好的权重
    sft_ckpt = "minimind_sft_final.pth"
    print(f"加载 SFT 基准权重: {sft_ckpt}")
    state_dict = torch.load(sft_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    ref_model.load_state_dict(state_dict)

    # 冻结参考模型，设为 eval 模式
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # --- 数据加载 ---
    dataset = DPODataset("./dataset-minimind/dpo_with_identity.jsonl", "./", max_length=512)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # --- 训练开始 ---
    model.train()
    global_step = 0
    epochs = 2

    print("开始 DPO 偏好对齐训练...")
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"DPO Epoch {epoch}")
        for step, batch in enumerate(pbar):
            # 将数据移动至设备
            c_ids, c_labels = batch['chosen_ids'].to(device), batch['chosen_labels'].to(device)
            r_ids, r_labels = batch['rejected_ids'].to(device), batch['rejected_labels'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # 1. 计算策略模型的 Logits (合并计算以节省显存)
                all_ids = torch.cat([c_ids, r_ids], dim=0)
                all_logits = model(all_ids)
                policy_c_logits, policy_r_logits = all_logits.chunk(2, dim=0)

                p_chosen_logps = get_batch_logps(policy_c_logits, c_labels)
                p_rejected_logps = get_batch_logps(policy_r_logits, r_labels)

                # 2. 计算参考模型的 Logits (不需要梯度)
                with torch.no_grad():
                    ref_all_logits = ref_model(all_ids)
                    ref_c_logits, ref_r_logits = ref_all_logits.chunk(2, dim=0)
                    r_chosen_logps = get_batch_logps(ref_c_logits, c_labels)
                    r_rejected_logps = get_batch_logps(ref_r_logits, r_labels)

                # 3. 计算 DPO 损失
                loss, reward_acc = dpo_loss(
                    p_chosen_logps, p_rejected_logps,
                    r_chosen_logps, r_rejected_logps,
                    beta=beta
                )

            # --- 反向传播 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 日志记录
            if step % 5 == 0:
                writer.add_scalar("Loss/DPO", loss.item(), global_step)
                writer.add_scalar("Metric/Reward_Acc", reward_acc.item(), global_step)
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "RewAcc": f"{reward_acc.item():.4f}"
                })

            global_step += 1

        # 保存每个 Epoch
        torch.save(model.state_dict(), f"checkpoint_dpo_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "minimind_dpo_final.pth")
    writer.close()
    print("DPO 训练完成。")



import os

if __name__ == "__main__":
    # 获取绝对路径
    tokenizer_abs_path = os.path.abspath("./")
    dataset_abs_path = os.path.abspath("./dataset-minimind/dpo_with_identity.jsonl")

    dataset = DPODataset(
        file_path=dataset_abs_path,
        tokenizer_path=tokenizer_abs_path,  # 使用绝对路径
        max_length=512
    )
    train_dpo()