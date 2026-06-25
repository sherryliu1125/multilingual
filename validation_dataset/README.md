---
license: apache-2.0
task_categories:
- text-classification
language: [en, th, id, ar, pt, es, tr, zh, ms]
tags:
- content-moderation
- hate-speech-detection
- multilingual
- low-resource
size_categories:
- 100K<n<1M
---
# 8 国小语种内容安全标注数据集 (Multilingual Content-Safety, 8 Countries)

覆盖 **8 个国家/地区**主流论坛的内容合规审核标注数据,聚焦**小语种 / 低资源语言**的仇恨言论、骚扰、危险内容等违规识别,并包含各国**本地文化红线**(如泰国王室、印尼 SARA、南非排外、土耳其辱国等)。

标注由 **5 模型 NIM 陪审团**(qwen3-next-80b / llama-3.3-70b / llama-4-maverick / nemotron-super-49b / gemma-3n)独立判断 + 多数表决(majority≥3/5, agreement≥0.8)产出。

## 目录结构

```
multiclass/<国家>/train.csv, test.csv   # 细分多分类(去泄漏切分, train/test 零重叠)
binary/<国家>.csv + all_8country.csv     # 二分类(0=safe, 1=violation)
ood/binary_ood.csv, multiclass_ood.csv   # 跨源 OOD 测试集(不同来源, 检验泛化)
augment/hard_negatives.csv               # 困难负样本(玩笑/自嘲/学术等带脏字但安全)
augment/quote_report_templated.csv       # 引用举报模板("我举报别人骂我"=safe)
```
所有 CSV 列:`text, label`(+ `country`)。

## 多分类数据(去泄漏 train_clean / test_clean)

| 国家 | 语种 | 训练 | 测试 | 类别数 | safe占比 |
|------|------|------|------|------|------|
| 新加坡 SG | 英/中/马来/泰米尔 | 37,114 | 6,551 | 8 | 45.9% |
| 泰国 TH | 泰语 | 19,566 | 3,452 | 8 | 25.1% |
| 印尼 ID | 印尼语 | 41,520 | 7,329 | 9 | 28.6% |
| 南非 ZA | 英/阿非利卡/祖鲁等 | 34,230 | 6,040 | 9 | 34.9% |
| 沙特 SA | 阿拉伯语 | 32,426 | 5,722 | 10 | 27.5% |
| 巴西 BR | 葡萄牙语(巴西) | 27,601 | 15,314 | 9 | 44.0% |
| 墨西哥 MX | 西班牙语(墨西哥) | 23,672 | 13,810 | 9 | 26.7% |
| 土耳其 TR | 土耳其语 | 28,003 | 15,314 | 9 | 46.9% |
| **合计** | 8 国 | **244,132** | **73,532** | — | — |

## 二分类数据(gold_binary)

| 国家 | 样本数 | 违规占比 |
|------|------|------|
| 新加坡 SG | 11,383 | 67.1% |
| 泰国 TH | 11,128 | 81.4% |
| 印尼 ID | 9,808 | 81.2% |
| 南非 ZA | 8,860 | 80.7% |
| 沙特 SA | 10,201 | 81.8% |
| 巴西 BR | 14,711 | 19.8% |
| 墨西哥 MX | 13,288 | 12.8% |
| 土耳其 TR | 14,861 | 39.7% |
| **合计** | **94,240** | 53.8% |

## OOD 跨源测试集

| 文件 | 样本数 | 说明 |
|------|------|------|
| ood/binary_ood.csv | 5,113 | 二分类, 与训练完全不同来源(consensus 标注) |
| ood/multiclass_ood.csv | 3,258 | 多分类, 跨源去噪 |

## 增强样本(边界鲁棒性)

| 文件 | 样本数 | 用途 |
|------|------|------|
| augment/hard_negatives.csv | 3,084 | — |
| augment/quote_report_templated.csv | 1,280 | — |

## 标签体系

**通用类别(所有国家):**
`safe` / `Hate_Speech` / `Harassment` / `Dangerous_Content` / `Sexually_Explicit_Information` / `Politically_Sensitive_Topics` / `Cybersecurity_Malware`

**各国本地红线(节选):**
SG_Racial_Religious_Harmony · TH_Lese_Majeste · ID_SARA · ID_Blasphemy · ZA_Xenophobia · ZA_Severe_Racism · SA_State_Security_Royalty · SA_Religious_Violation · SA_LGBTQ_Content · BR_Political_Extremism · BR_Structural_Racism · MX_Narco_Culture · MX_Gender_Violence · TR_Insulting_State · TR_Separatism_Terror

完整标签集(22 类):
`BR_Political_Extremism, BR_Structural_Racism, Cybersecurity_Malware, Dangerous_Content, Harassment, Hate_Speech, ID_Blasphemy, ID_SARA, MX_Gender_Violence, MX_Narco_Culture, Politically_Sensitive_Topics, SA_LGBTQ_Content, SA_Religious_Violation, SA_State_Security_Royalty, SG_Racial_Religious_Harmony, Sexually_Explicit_Information, TH_Lese_Majeste, TR_Insulting_State, TR_Separatism_Terror, ZA_Severe_Racism, ZA_Xenophobia, safe`

## ⚠️ 关于标签质量(重要)

标签由 LLM 陪审团自动标注,经独立强模型审计发现:模型与标签不一致的样本中,约 **64% 其实是标签错**(模型判断更准)。因此:
- 在本数据**原始标签**上训练的多分类模型 macro-F1 ≈ 0.70,
- 但用独立强模型清洗错标签后,**真实 F1 ≈ 0.88**。

即:**标签是当前性能天花板**,数据本身信息量足够支撑 ~0.88 的多分类精度。使用时建议结合置信学习 / 标签清洗。

## 配套模型

已在 ModelScope 发布基于本数据训练的模型:
- `minions0213/content-safety-binary-xlmr-final` — 最强二分类(gold F1 0.941)
- `minions0213/content-safety-binary-mdeberta-final` — 轻量二分类(560MB)
- `minions0213/content-safety-multiclass-percountry-lora` — 每国独立 LoRA 多分类

## 用途
内容审核 / 仇恨言论检测研究 / 低资源语言安全模型训练 / 标签噪声与置信学习研究。

## 引用
如使用本数据,请注明来源。标注方法:5 模型 NIM 陪审团 + 多数表决。
