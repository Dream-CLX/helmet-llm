#微调后模型 test 集批量推理
import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# =========================
# 路径配置
# =========================
BASE_MODEL_PATH = "/root/autodl-tmp/helmet_llm/models/Qwen2.5-VL-3B-Instruct"
LORA_MODEL_PATH = "/root/autodl-tmp/helmet_llm/output/qwen25vl_shwd_lora_full"

TEST_JSONL = "/root/autodl-tmp/helmet_llm/data/shwd_helmet_test.jsonl"
DATA_ROOT = "/root/autodl-tmp/helmet_llm/data"
OUTPUT_JSONL = "/root/autodl-tmp/helmet_llm/output/test_pred_lora.jsonl"

MAX_NEW_TOKENS = 256
MAX_SAMPLES = None   # 如果想先测试 100 条，可以改成 100


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def infer_one(model, processor, image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image_path],
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def main():
    test_items = load_jsonl(TEST_JSONL)

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    processor = AutoProcessor.from_pretrained(LORA_MODEL_PATH)

    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for idx, item in enumerate(test_items, 1):
            image_rel = item["images"][0]
            image_path = str(Path(DATA_ROOT) / image_rel)
            prompt = item.get("instruction", "").strip()

            pred = infer_one(model, processor, image_path, prompt)

            out_item = {
                "image": image_rel,
                "instruction": prompt,
                "pred_output": pred
            }
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")

            print(f"[lora] {idx}/{len(test_items)} done: {image_rel}")

            if MAX_SAMPLES is not None and idx >= MAX_SAMPLES:
                break

    print(f"\n微调后模型预测完成，结果保存在: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()