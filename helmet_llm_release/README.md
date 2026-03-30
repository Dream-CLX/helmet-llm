# Helmet-LLM：基于 Qwen2.5-VL 的安全帽佩戴风险识别微调项目

## 项目简介

本项目基于 **Qwen2.5-VL-3B-Instruct** 多模态大模型，结合 **LLaMA-Factory** 框架，采用 **LoRA（Low-Rank Adaptation）** 方式完成安全帽佩戴风险识别任务的指令微调，并提供了从 **数据准备**、**模型训练**、**微调前后推理** 到 **效果评估** 的完整实验流程。

本项目的核心目标是：  
给定一张施工场景图像和一条任务指令，让模型能够输出关于安全帽佩戴情况的自然语言分析结果，包括：

- 是否存在安全帽佩戴风险
- 风险等级
- 风险原因
- 整改建议

项目最终发布的是 **LoRA 适配器权重**，而不是完整微调模型。使用时需要先准备基础模型，再将 LoRA 权重挂载到基础模型上进行推理。

---

## 效果展示
 

## 项目特点

- 基于 **Qwen2.5-VL-3B-Instruct** 进行多模态指令微调
- 使用 **LLaMA-Factory** 进行 LoRA 训练
- 支持 **微调前 / 微调后** 模型推理对比
- 支持从文本中抽取结构化任务结果并进行评估
- 提供 **分类指标 + 文本生成指标 + 任务约束指标** 的综合评估方案
- 项目结构清晰，适合复现、展示与二次开发

---

## 项目结构

```text
helmet_llm/
├── README.md
├── requirements.txt
├── train_qwen25vl_shwd_full.yaml
├── scripts/
│   ├── base_infertest.py
│   ├── finetuning_infertest.py
│   ├── evaluate_compare.py
│   └── check_loadsuccess.py
├── data/
│   ├── dataset_info.json
│   ├── shwd_helmet_train.jsonl
│   ├── shwd_helmet_val.jsonl
│   └── shwd_helmet_test.jsonl
├── lora/
│   └── qwen25vl_shwd_lora_full/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── added_tokens.json
│       ├── chat_template.jinja
│       ├── merges.txt
│       ├── preprocessor_config.json
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── video_preprocessor_config.json
│       └── vocab.json
```
---

## 环境配置
```bash
conda create -n qwen25vl_lf python=3.11 -y
conda activate qwen25vl_lf
python -m pip install -U pip setuptools wheel
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cd /autodl-tmp/LLaMA-Factory
pip install -e .
pip install -r requirements/metrics.txt
```



## 微调模型说明
本项目使用的基础模型为：Qwen2.5-VL-3B-Instruct

由于基础模型体积较大，本仓库 不直接提供基础模型权重。请使用者自行下载，并放置到本地目录，例如：models/Qwen2.5-VL-3B-Instruct/。关于它的下载可以在hugging-face开源项目中找到

---

## 数据集说明

本项目使用安全帽佩戴风险识别相关数据进行微调。仓库中提供了训练、验证、测试所需的 JSONL 标注文件，但不直接提供原始图片文件。

当前仓库中提供的内容
```text
data/
├── dataset_info.json
├── shwd_helmet_train.jsonl
├── shwd_helmet_val.jsonl
└── shwd_helmet_test.jsonl
```
图片数据获取方式:https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset?tab=readme-ov-file,请自行下载图像数据，并将图片整理到如下目录中：
```text
data/
└── JPEGImages/
```

---

## 训练方法

本项目使用 LLaMA-Factory 进行训练，采用 LoRA 微调方式。

###训练配置文件
```text
train_qwen25vl_shwd_full.yaml
```
###训练命令
```bash
llamafactory-cli train train_qwen25vl_shwd_full.yaml
```

训练配置说明,训练配置文件中通常包含以下内容：

- 基础模型路径
- 微调方式（LoRA）
- 数据集名称与路径
- 模板类型（如 qwen2_vl）
- batch size、学习率、epoch 等训练参数
- 输出目录配置

---

## 测试方法
### 微调前模型推理
目中提供了微调前模型在测试集上的批量推理脚本：
```text
scripts/base_infertest.py
```
运行方式：
```text
python scripts/base_infertest.py
```
### 项目中提供了微调后模型在测试集上的批量推理脚本：
```text
scripts/finetuning_infertest.py
```
运行方式：
```text
python scripts/finetuning_infertest.py
```

### 评估方法

为了比较 微调前模型 与 微调后模型 的效果，本项目提供了评估脚本：
```text
scripts/evaluate_compare.py
```
运行方式：
```text
python scripts/evaluate_compare.py
```
评估指标,本项目的评估分为三类：

1. 结构化任务指标,从模型生成文本中抽取结构化信息，进行分类评估：
- 风险存在判断
- Accuracy
- Precision
- Recall
- F1
- Coverage

风险等级判断
- Accuracy
- Macro-F1
- Coverage

2. 文本生成指标
衡量生成文本与参考答案之间的相似程度：
- ROUGE-L
- BLEU
- BERTScore

3. 任务约束指标
衡量输出是否符合任务要求：
off-topic rate：是否出现明显跑题内容
format completeness：回答是否包含“风险判断 + 风险等级 + 原因 + 建议”等关键信息

