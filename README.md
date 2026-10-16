[English](./README.en.md) | **简体中文**

<div align="center">

# 去腔 · zhsteer

**把中文 AI 腔从 Qwen3 权重里投影掉 —— 一个推理时消融原语，面向中文内容管线与 Agent 流程。**

[![CI](https://img.shields.io/github/actions/workflow/status/SuperMarioYL/zhsteer/ci.yml?branch=main&label=CI&logo=github)](https://github.com/SuperMarioYL/zhsteer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-%E2%89%A516GB%20VRAM-76B900?logo=nvidia&logoColor=white)](#安装)
[![模型](https://img.shields.io/badge/%E5%B7%B2%E9%AA%8C%E8%AF%81-Qwen3--4B--Instruct--2507-6F42C1)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

![typing](https://readme-typing-svg.demolab.com?font=Noto+Sans+SC&weight=600&size=20&duration=3200&pause=1000&color=58A6FF&center=true&vCenter=true&width=680&height=60&lines=%E4%BB%8E%20Qwen3%20%E6%AE%8B%E5%B7%AE%E6%B5%81%E6%8F%90%E5%8F%96%E4%B8%AD%E6%96%87%20AI%20%E8%85%94%E6%96%B9%E5%90%91%3B%E6%8E%A8%E7%90%86%E6%97%B6%E6%8A%95%E5%BD%B1%E6%B6%88%E8%9E%8D%20x%20%E2%86%90%20x%20%E2%88%92%20%28x%C2%B7d%CC%82%29d%CC%82%3B%E6%8E%92%E6%AF%94%E4%B8%89%E8%BF%9E%E3%80%81%E5%80%BC%E5%BE%97%E6%B3%A8%E6%84%8F%E7%9A%84%E6%98%AF%E3%80%81%E7%88%86%E6%AC%BE%E4%BD%93%3B%E5%9C%A8%E7%94%9F%E6%88%90%E6%BA%90%E5%A4%B4%E6%B6%88%E5%A4%B1%EF%BC%8C%E8%80%8C%E4%B8%8D%E6%98%AF%E5%9C%A8%E7%BC%96%E8%BE%91%E5%B1%82%E8%BF%94%E5%B7%A5)

```bash
uv run zhsteer demo --model qwen3-4b     # → assets/demo/ablation-report.md
```

免费 + 开源 · MIT · 无付费层

</div>

## 它解决什么问题

跑着 Qwen/GLM/DeepSeek 的中文内容管线，每天都在和模型自带的 AI 腔打交道：排比三连、“值得注意的是”、“随着……的不断发展”、爆款体。现成的解法是把去 AI 腔做成 prompt 层 Skill（blader/humanizer、Humanizer-zh 这一系），在编辑层逐篇重写。两条路都是在对付症状——下一轮 `generate()`，腔调重新长出来；每换一次模型版本，重写一遍规则库。

zhsteer 把同一件事挪到权重层：

1. 用 64 组手写中文最小对（同一段内容的 AI 腔写法 vs 朴素写法）喂给 Qwen3-4B，在残差流上做差值均值（diff-in-means），得到一个“中文 AI 腔方向”；
2. 生成时把这个方向从残差流里投影掉：`x ← x − (x·d̂)d̂`。腔调在生成源头就不再进入输出，与之正交的内容子空间不受影响；
3. 它交付为一个可调用的推理原语——`steer()` 上下文管理器或 `--ablate 0.9`，像 `temperature` 一样接在你现有的 transformers 调用旁边。方向提取每个模型版本只做一次，Agent 内容管线不必逐篇重洗。

## 先读这段：v0.1 的诚实边界

- **只在 Qwen/Qwen3-4B-Instruct-2507 上验证过。** DirectionPack 按模型打标并校验 checksum，拿到别的模型上会拒绝加载。
- **“中文输出的方向可分性”正是 v0.1 要跑通的证伪实验。** 计划设定的证伪线：held-out 最小对可分性 ≥ 90%，平均 trope 分数降幅 ≥ 20%，且无退化输出。这些数字应当来自你自己的 GPU——跑一遍 `zhsteer demo`，读报告。不达标？负结果本身也值得发布，这正是本项目把报告设为核心产物的原因。
- **仓库内置的 [`assets/demo/ablation-report.md`](./assets/demo/ablation-report.md) 是报告模板**，结构完整、数值全部为占位符，不是实测结果。真实数据由你的 `zhsteer demo` 运行写回该路径。

## 安装

要求：NVIDIA GPU（≥ 16GB 显存）、Python 3.12+、[uv](https://docs.astral.sh/uv/)。macOS / CPU 推理不在 v0.1 支持范围。

```bash
git clone https://github.com/SuperMarioYL/zhsteer
cd zhsteer
uv sync
uv run zhsteer doctor    # 体检：CUDA、显存、模型下载磁盘
```

显存不够或没有 CUDA？`doctor` 会逐项列出缺什么，不会让你白等一次 8GB 下载。

## 快速开始

```bash
# 一条命令：下载模型（首次 ~8GB）→ 提取方向 → 重新生成 10 条 held-out 提示词 → 前后打分 → 写报告
# 预计 ~30 分钟（16GB 消费级 GPU，模型只需下载一次）
uv run zhsteer demo --model qwen3-4b
```

想分步执行也可以：

```bash
uv run zhsteer extract --model qwen3-4b            # pairs/seed_pairs.zh.yaml → packs/qwen3-4b.pack
uv run zhsteer generate --pack packs/qwen3-4b.pack \
    --ablate 0.9 --prompt "写一段产品介绍"           # 前后对照 + trope 分数
```

## 演示报告

`zhsteer demo` 的产物是一份消融报告：每条提示词的 Before/After 全文、确定性 trope 分数、退化检查，以及汇总表。**[查看当前仓库内置的报告模板 →](./assets/demo/ablation-report.md)**（占位符结构，跑完 demo 后该文件会被你的实测数据覆盖）。报告长这样：

```markdown
# zhsteer 消融报告
- 模型：Qwen/Qwen3-4B-Instruct-2507
- 方向包：packs/qwen3-4b.pack（layer <L>，checksum <…>）
- 最小对：pairs/seed_pairs.zh.yaml（train 52 / held-out 12）

## 各提示词前后对比
### 1. <提示词摘要>
**Before（原始）**                    **After（消融，强度 1.0）**
<模型原始输出，腔调完整>               <投影消融后的输出>

trope 分数：<XX.XX> → <XX.XX>（降幅 <XX%>）；退化检查：通过

## 分数汇总
| # | 提示词 | Before | After | 降幅 |
| 1 | <…>   | <XX.XX> | <XX.XX> | <XX%> |
平均 trope 分数：<XX.XX> → <XX.XX>（平均降幅 <XX%>）
```

打分器是纯词表 + 正则的确定性实现——同一文本永远得到同一分数，前后差值只能来自消融本身。

## 工作原理

```text
zhsteer CLI (typer)
 ├─ loader.py   — 模型加载：CUDA / bf16（生成钩子归 ablate 管）
 ├─ extract.py  — 最小对激活（逐层均值池化）→ diff-in-means 方向
 ├─ ablate.py   — 生成时投影消融 x ← x − (x·d̂)d̂，强度可调
 ├─ score.py    — 确定性 trope 打分器（排比三连 / 值得体 / 爆款体等词表）
 └─ pack.py     — DirectionPack 保存 / 加载 / 校验（schema v1，带校验和）

pairs/seed_pairs.zh.yaml ──extract──▶ packs/qwen3-4b.pack ──steer()──▶ 消融输出 ──score──▶ ablation-report.md
```

核心数据结构是 **DirectionPack**：一个带 sha256 校验和、按模型打标的 artifact，内含方向向量本身和提取时的验证证据（逐层可分性、train/held-out 拆分、样本数）。加载时校验和与字段会被重新计算——手改过的或损坏的 pack 会当场报错，而不是带着坏方向继续生成。

层约定：方向记录在“第 *i* 层残差流之后”，与消融钩子挂在解码器第 *i* 层输出上是同一个点，提取的层直接用于消融。方向默认选训练集可分性最高的层（平手取更深），held-out 可分性只作为诚实的泛化数字写进 pack，不参与选择。

## 用法

**接进你自己的管线**（3 行核心代码，完整示例见 [`examples/steer_demo.py`](./examples/steer_demo.py)）：

```python
from zhsteer import load, steer

pack = load("packs/qwen3-4b.pack")
with steer(model, pack, strength=0.9):
    output = model.generate(**inputs)   # 你现有的 transformers 调用，一行不改
```

`strength=1.0` 恰好移除腔调分量；调小则部分抑制，调大则过移除。钩子在上下文进入时挂载、退出时移除，块内抛异常也不会泄漏到外层生成。

**命令一览**：

| 命令 | 作用 |
| --- | --- |
| `zhsteer doctor` | 体检 CUDA、≥ 16GB 显存、模型下载磁盘空间 |
| `zhsteer extract --model qwen3-4b` | 从内置 64 组最小对提取方向，写 `packs/qwen3-4b.pack` |
| `zhsteer generate --pack <pack> --prompt <…> --ablate 0.9` | 同一提示词生成两遍，前后对照输出 + 分数 |
| `zhsteer demo --model qwen3-4b` | 一条命令跑完全链路并写消融报告 |
| `zhsteer score --text <…>` | 单独用确定性 trope 打分器给一段文本打分（不需要模型） |

自备最小对？`--pairs your_pairs.yaml` 传你自己的集合，schema 与内置文件一致（`ai` / `plain` / `held_out`）。

## 打分口径

打分器覆盖九类中文 AI 腔信号：排比三连、值得体（“值得注意的是”家族）、总结套话、空话连接、递进句式、万能开头、首先其次体、爆款体词表、感叹号密度。分数是每 1000 非空白字符的加权命中数——密度口径让不同长度的 Before/After 可比。输出为空或整段退化属于质量失败，由独立的退化检查（成段重复检测）负责，不算进腔调分数。

## 已知局限

- v0.1 只验证 Qwen3-4B-Instruct-2507 一个模型；换模型需重新提取 pack（schema 已按模型打标，不会静默错配）。
- 仅支持 HF transformers 推理循环；vLLM / SGLang / OpenAI 兼容 serving 集成未实现。
- 单方向、单层消融；风格 × 格式 × 语言的多方向组合不在 v0.1。
- 打分器是词表口径的确定性评分，不是 LLM-as-judge 的质量评审——它度量腔调密度，不度量文采。
- 种子最小对为手工编写，覆盖常见内容场景但非穷举；自定义场景请用 `--pairs` 换成你自己的集合。

## 路线图

以下均为**未实现**的计划项：

- [ ] GLM-4 / DeepSeek 的方向包验证（schema 已就绪）
- [ ] vLLM / SGLang serving 集成，把 `--ablate` 变成一行 serving 参数
- [ ] 多方向组合（风格 × 格式 × 语言）
- [ ] 方向包发布流程：每个模型版本 bump 附带新报告数字

## 变更记录

见 [CHANGELOG.md](./CHANGELOG.md)。

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
