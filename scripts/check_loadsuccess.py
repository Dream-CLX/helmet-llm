#测试模型是否下载成功并且可以成功导入图片
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

MODEL_PATH = "/root/autodl-tmp/helmet_llm/models/Qwen2.5-VL-3B-Instruct"
IMAGE_PATH = "/root/autodl-tmp/helmet_llm/data/JPEGImages/000012.jpg"

PROMPT = "请判断这张图片是否存在安全帽佩戴风险，并给出风险等级、原因和建议。"

def main():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs = [IMAGE_PATH]

    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("===== PROMPT =====")
    print(PROMPT)
    print("\n===== RESPONSE =====")
    print(output_text[0])

if __name__ == "__main__":
    main()