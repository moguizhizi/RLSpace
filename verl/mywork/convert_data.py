import numpy as np
import pandas as pd
import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_PATH = "Models/Qwen3-8B-SFT-Merged-TFG/0121"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 1. 定义你的 Prompt 模板 (保持和原代码一致)
SYSTEM_MESSAGE = (
    "You are an expert clinical doctor. Your task is to recommend a comprehensive list of medications for the patient.\n\n"
    "### INSTRUCTIONS:\n"
    "1. **Analyze**: First, generate an **internal clinical reasoning** enclosed in `<think>...</think>` tags.\n"
    "2. **Structure**: Group your reasoning by medical condition or organ system (e.g., 'Cardiovascular Management').\n"
    "3. **Prescribe**: Finally, output the recommended medications in a Python list format."
)


def make_dataset(json_path, output_parquet_path):
    # 读取原始 JSON
    ds = load_dataset("json", data_files=json_path, split="train")

    data_list = []
    token_counts = []  # 用于存储每条数据的长度
    for example in ds:
        # 构造对话 Prompt
        prompt = (
            f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n## Patient Info:\n{example['input']}\n\nPlease recommend medications.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        ids = tokenizer.encode(prompt, add_special_tokens=False)
        current_len = len(ids)
        token_counts.append(current_len)

        # VeRL 数据协议:
        # data_source: 数据来源标记
        # prompt: 输入给模型的文本 (包含 prompt template)
        # ability: 任务类型 (可选)
        # reward_model: 存放 Ground Truth，用于传给奖励函数
        # extra_info: 存放原始 input，用于正则提取 Candidate

        data_list.append(
            {
                "data_source": "mimic-iii",
                "prompt": [
                    {"role": "user", "content": prompt}
                ],  # VeRL 接收 chat list 或 string，这里存 raw string 更好处理
                "prompt_raw": prompt,  # 我们直接用 raw string 喂给 vLLM
                "ability": "medical",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example["output"],
                },
                "extra_info": {
                    "original_input": example[
                        "input"
                    ]  # 需要保留原始 input 来提取 candidates
                },
            }
        )

    df = pd.DataFrame(data_list)

    max_len = np.max(token_counts)
    min_len = np.min(token_counts)
    avg_len = np.mean(token_counts)
    p95 = np.percentile(token_counts, 95)
    p99 = np.percentile(token_counts, 99)

    print(f"📊 统计结果 [{output_parquet_path}]:")
    print(f"   - 总数据量: {len(token_counts)}")
    print(f"   - Max Length (最大值): {max_len}")
    print(f"   - Min Length (最小值): {min_len}")
    print(f"   - Avg Length (平均值): {avg_len:.2f}")
    print(f"   - P95 Length (95%数据小于): {p95:.2f}")
    print(f"   - P99 Length (99%数据小于): {p99:.2f}")

    # 转换为 Parquet
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df.to_parquet(output_parquet_path)
    print(f"✅ 已转换 {len(df)} 条数据至: {output_parquet_path}")

    return df


# 执行转换
if __name__ == "__main__":
    d1 = make_dataset(
        "MR/data_think/output/TFG1/test_sft.json", "verl/mywork/data/test.parquet"
    )
    d2 = make_dataset(
        "MR/data_think/output/TFG1/train_sft.json", "verl/mywork/data/train.parquet"
    )
    d3 = make_dataset(
        "MR/data_think/output/TFG1/val_sft.json", "verl/mywork/data/val.parquet"
    )
    print(len(d1), len(d2), len(d3))
    d_val = d3[:100]
    d_train = pd.concat([d2, d3[100:]], ignore_index=True)
    d_test = d1
    print(len(d_train), len(d_val), len(d_test))
    d_train.to_parquet("verl/mywork/data/train.parquet")
    d_val.to_parquet("verl/mywork/data/val.parquet")
    d_test.to_parquet("verl/mywork/data/test.parquet")
    print("Done!")
