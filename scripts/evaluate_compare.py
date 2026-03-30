#微调前后模型对测试集指标对比（人为选择指标）
import json
import re
from pathlib import Path
from statistics import mean

import jieba
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore_score


# =========================
# 1. 文件路径
# =========================
TEST_JSONL = "/root/autodl-tmp/helmet_llm/data/shwd_helmet_test.jsonl"
BASE_PRED_JSONL = "/root/autodl-tmp/helmet_llm/output/test_pred_base.jsonl"
LORA_PRED_JSONL = "/root/autodl-tmp/helmet_llm/output/test_pred_lora.jsonl"

OUTPUT_CSV = "/root/autodl-tmp/helmet_llm/output/eval_compare_metrics.csv"
OUTPUT_DETAIL_JSONL = "/root/autodl-tmp/helmet_llm/output/eval_compare_details.jsonl"


# =========================
# 2. 文本解析规则
# =========================
OFF_TOPIC_KEYWORDS = [
    "材质", "护目镜", "认证", "国家标准", "安全标准", "头盔完整", "完整视图",
    "高空坠落", "机械碰撞", "火灾", "坍塌"
]

RISK_TRUE_PATTERNS = [
    r"存在.*风险",
    r"存在安全帽.*风险",
    r"发现.*未佩戴安全帽",
    r"未佩戴安全帽.*风险",
    r"佩戴违规",
]

RISK_FALSE_PATTERNS = [
    r"未发现.*风险",
    r"不存在.*风险",
    r"未检测到.*未佩戴安全帽",
    r"未发现明显安全帽佩戴风险",
]

LOW_PATTERNS = [r"风险等级为低", r"低风险"]
MEDIUM_PATTERNS = [r"风险等级为中", r"中风险"]
HIGH_PATTERNS = [r"风险等级为高", r"高风险"]


# =========================
# 3. 工具函数
# =========================
def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize_image_key(image_value):
    if image_value is None:
        return None
    return str(image_value).replace("\\", "/")


def get_pred_text(item):
    # 兼容不同字段名
    for key in ["pred_output", "response", "output", "answer"]:
        if key in item and isinstance(item[key], str):
            return item[key].strip()
    return ""


def extract_risk_present(text):
    text = text.strip()

    # 先匹配 false，避免“存在风险”被“不存在风险”误伤
    for p in RISK_FALSE_PATTERNS:
        if re.search(p, text):
            return 0

    for p in RISK_TRUE_PATTERNS:
        if re.search(p, text):
            return 1

    return None


def extract_risk_level(text):
    text = text.strip()

    for p in HIGH_PATTERNS:
        if re.search(p, text):
            return "high"

    for p in MEDIUM_PATTERNS:
        if re.search(p, text):
            return "medium"

    for p in LOW_PATTERNS:
        if re.search(p, text):
            return "low"

    return None


def has_reason(text):
    keywords = ["原因", "由于", "因为", "系", "未佩戴", "违规", "风险依据"]
    return any(k in text for k in keywords)


def has_advice(text):
    keywords = ["建议", "应", "立即", "整改", "规范佩戴", "持续保持"]
    return any(k in text for k in keywords)


def format_completeness(text):
    # 是否包含：风险存在判断 + 风险等级 + 原因 + 建议
    score = 0

    if extract_risk_present(text) is not None:
        score += 1
    if extract_risk_level(text) is not None:
        score += 1
    if has_reason(text):
        score += 1
    if has_advice(text):
        score += 1

    return score / 4.0


def off_topic_rate(text):
    return 1.0 if any(k in text for k in OFF_TOPIC_KEYWORDS) else 0.0


def calc_rouge_l(preds, refs):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for pred, ref in zip(preds, refs):
        s = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(s)
    return mean(scores) if scores else 0.0


def tokenize_zh(text):
    return list(jieba.cut(text))


def calc_bleu(preds, refs):
    smoothie = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(preds, refs):
        pred_tokens = tokenize_zh(pred)
        ref_tokens = tokenize_zh(ref)
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            scores.append(0.0)
            continue
        score = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            smoothing_function=smoothie
        )
        scores.append(score)
    return mean(scores) if scores else 0.0


def calc_bertscore(preds, refs):
    if len(preds) == 0:
        return 0.0
    _, _, f1 = bertscore_score(preds, refs, lang="zh", verbose=False)
    return float(f1.mean().item())


def safe_metric_binary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def safe_metric_multiclass(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


# =========================
# 4. 构建 gold 映射
# =========================
def build_gold_map(test_items):
    gold_map = {}

    for item in test_items:
        image = normalize_image_key(item["images"][0])
        meta = item.get("meta", {})
        gold_map[image] = {
            "gold_output": item.get("output", ""),
            "gold_risk_present": int(meta.get("risk_present")) if "risk_present" in meta else None,
            "gold_risk_level": meta.get("risk_level"),
            "meta": meta,
        }

    return gold_map


# =========================
# 5. 评估单个预测文件
# =========================
def evaluate_prediction_file(pred_items, gold_map, model_name):
    details = []

    for item in pred_items:
        image = normalize_image_key(item.get("image"))
        if image not in gold_map:
            continue

        pred_text = get_pred_text(item)
        gold_info = gold_map[image]

        pred_risk_present = extract_risk_present(pred_text)
        pred_risk_level = extract_risk_level(pred_text)

        detail = {
            "model_name": model_name,
            "image": image,
            "gold_output": gold_info["gold_output"],
            "pred_output": pred_text,
            "gold_risk_present": gold_info["gold_risk_present"],
            "pred_risk_present": pred_risk_present,
            "gold_risk_level": gold_info["gold_risk_level"],
            "pred_risk_level": pred_risk_level,
            "off_topic": off_topic_rate(pred_text),
            "format_completeness": format_completeness(pred_text),
        }
        details.append(detail)

    # ===== 结构化指标 =====
    # 只统计成功抽取出标签的样本
    rp_valid = [d for d in details if d["pred_risk_present"] is not None and d["gold_risk_present"] is not None]
    rl_valid = [d for d in details if d["pred_risk_level"] is not None and d["gold_risk_level"] is not None]

    metrics = {"model_name": model_name}

    if len(rp_valid) > 0:
        y_true = [d["gold_risk_present"] for d in rp_valid]
        y_pred = [d["pred_risk_present"] for d in rp_valid]
        binary_metrics = safe_metric_binary(y_true, y_pred)
        metrics["risk_present_accuracy"] = binary_metrics["accuracy"]
        metrics["risk_present_precision"] = binary_metrics["precision"]
        metrics["risk_present_recall"] = binary_metrics["recall"]
        metrics["risk_present_f1"] = binary_metrics["f1"]
        metrics["risk_present_coverage"] = len(rp_valid) / len(details) if details else 0.0
    else:
        metrics["risk_present_accuracy"] = None
        metrics["risk_present_precision"] = None
        metrics["risk_present_recall"] = None
        metrics["risk_present_f1"] = None
        metrics["risk_present_coverage"] = 0.0

    if len(rl_valid) > 0:
        y_true = [d["gold_risk_level"] for d in rl_valid]
        y_pred = [d["pred_risk_level"] for d in rl_valid]
        multi_metrics = safe_metric_multiclass(y_true, y_pred)
        metrics["risk_level_accuracy"] = multi_metrics["accuracy"]
        metrics["risk_level_macro_f1"] = multi_metrics["macro_f1"]
        metrics["risk_level_coverage"] = len(rl_valid) / len(details) if details else 0.0
    else:
        metrics["risk_level_accuracy"] = None
        metrics["risk_level_macro_f1"] = None
        metrics["risk_level_coverage"] = 0.0

    # ===== 文本生成指标 =====
    preds = [d["pred_output"] for d in details]
    refs = [d["gold_output"] for d in details]

    metrics["rougeL_f1"] = calc_rouge_l(preds, refs)
    metrics["bleu"] = calc_bleu(preds, refs)
    metrics["bertscore_f1"] = calc_bertscore(preds, refs)

    # ===== 任务约束指标 =====
    metrics["off_topic_rate"] = mean([d["off_topic"] for d in details]) if details else 0.0
    metrics["format_completeness"] = mean([d["format_completeness"] for d in details]) if details else 0.0

    return metrics, details


# =========================
# 6. 主程序
# =========================
def main():
    test_items = load_jsonl(TEST_JSONL)
    base_pred_items = load_jsonl(BASE_PRED_JSONL)
    lora_pred_items = load_jsonl(LORA_PRED_JSONL)

    gold_map = build_gold_map(test_items)

    base_metrics, base_details = evaluate_prediction_file(base_pred_items, gold_map, "base_model")
    lora_metrics, lora_details = evaluate_prediction_file(lora_pred_items, gold_map, "lora_model")

    # 保存总指标
    df = pd.DataFrame([base_metrics, lora_metrics])
    print("\n===== 总体指标对比 =====")
    print(df.T)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n总体指标已保存到: {OUTPUT_CSV}")

    # 保存逐条明细
    with open(OUTPUT_DETAIL_JSONL, "w", encoding="utf-8") as f:
        for item in base_details + lora_details:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"逐条对比明细已保存到: {OUTPUT_DETAIL_JSONL}")


if __name__ == "__main__":
    main()