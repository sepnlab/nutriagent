# Data and code access

This repository is the public code and query release for the NutriAgent
manuscript (v1.0.1). It is not a full data release. This file maps the
reproducibility items requested in the R4.10 review comment to their access
status.

## Public in this repository

| Item | Location |
|---|---|
| Source code | `src/` |
| System, specialist, router, and synthesis prompts | embedded in `src/agent/` |
| 54-query curated set | `queries/layer/evaluation_queries_curated.json` and `curated_queries.txt` |
| 195-query candidate pool | `queries/layer/evaluation_queries_pool195.json` and `pool195_queries.txt` |
| Architecture, synthesis, and ablation configurations | `config/` |
| Evaluation code | `src/evaluation/` |
| Model IDs | `config/models.yaml` |

Model access and price date: **2026-10-20**. The reported model IDs are:

| Role | Provider | Model ID |
|---|---|---|
| NutriAgent backbone; DeepSeek-V4-Flash baseline | SiliconFlow | `deepseek-ai/DeepSeek-V4-Flash` |
| DeepSeek-V4-Pro baseline; robustness evaluator | DeepSeek | `deepseek-v4-pro` |
| Qwen3-8B baseline | SiliconFlow | `Qwen/Qwen3-8B` |
| Qwen3.8-Flash baseline | Qwen AI Platform | `qwen3.8-flash` |
| GLM-4.5-Air baseline | SiliconFlow | `zai-org/GLM-4.5-Air` |
| MiniMax-M2.5 baseline | Tencent TokenHub | `minimax-m2.5` |
| Qwen3.8-27B D6 backbone | Qwen AI Platform | `qwen3.8-27b` |
| MiMo-V2.5-Pro primary evaluator | Xiaomi | `mimo-v2.5-pro` |
| Local embedding | local | `Qwen3-Embedding-0.6B` |
| Local NLI evaluator | local | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |

Per-model list prices are in Supplementary Table S9a. Numeric expected values
are included in the query JSON where available: 36 of 54 curated queries and
3 of 195 candidate-pool queries. The plain-text files contain the question text
and query ID only.

## Controlled access

The following materials are available to the editor and reviewers on reasonable
request, subject to confidentiality, privacy, and source-licensing conditions:

- full answer keys and expected-behavior annotations;
- de-identified per-query expert scores;
- audit codebooks and de-identified audit records;
- row-level provenance records underlying Supplementary Table S3, where the
  original source terms permit sharing.

## Not redistributed

- The layer-hen corpus, because some source terms do not clearly permit
  redistribution. Dataset-level provenance and unresolved fields are reported
  in Supplementary Table S3.
- Raw expert workbooks and expert identifiers, which contain identifying
  information.
- Name/label maps, coding keys, API keys, logs, internal audit reports, and
  generated annotation content.
- The 24-query external diagnostic set, which requires permission from its
  external authors.
- The internal regression test suite, which depends on the non-redistributed
  corpus.
