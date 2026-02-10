import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from TransformerLM import TransformerLM


def stream_chat_pro():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)

    # 2. 初始化模型
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

    # 加载 SFT 权重
    model_path = r"sft_pre_rmsnorm_weight/minimind_sft_final.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 采样参数设置 ---
    temperature = 0.7  # 控制随机度：越低越保守，越高越有创意
    top_k = 50  # 只从概率最高的前 K 个 token 中采样
    top_p = 0.9  # 核采样：只从累积概率达到 P 的 token 集合中采样
    repetition_penalty = 1.2  # 重复惩罚：降低已出现词的权重

    print(f"\n--- MiniMind Pro 推理已就绪 (Temp={temperature}, Top_P={top_p}) ---")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['quit', 'exit']: break

        prompt = f"User: {user_input}Assistant: "
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        curr_input = torch.tensor([input_ids]).to(device)

        print("Assistant: ", end="", flush=True)

        generated_ids = input_ids
        for _ in range(256):
            with torch.no_grad():
                logits = model(curr_input)
                next_token_logits = logits[:, -1, :] / temperature

                # 应用重复惩罚 (简单版)
                for token_id in set(generated_ids):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty

                # Top-K 过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Top-P (Nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[0, indices_to_remove] = -float('Inf')

                # 最终采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()

            if next_token_id == tokenizer.eos_token_id: break

            word = tokenizer.decode([next_token_id], skip_special_tokens=True)
            print(word, end="", flush=True)

            generated_ids.append(next_token_id)
            curr_input = torch.tensor([generated_ids]).to(device)
            if curr_input.size(1) >= 512: break
        print()


if __name__ == "__main__":
    stream_chat_pro()