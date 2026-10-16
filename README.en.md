**English** | [简体中文](./README.md)

<div align="center">

# zhsteer

**Project the Chinese AI register out of Qwen3's weights — an inference-time ablation primitive for Chinese content pipelines and Agent flows.**

[![CI](https://img.shields.io/github/actions/workflow/status/SuperMarioYL/zhsteer/ci.yml?branch=main&label=CI&logo=github)](https://github.com/SuperMarioYL/zhsteer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![CUDA](https://img.shields.io/badge/CUDA-%E2%89%A516GB%20VRAM-76B900?logo=nvidia&logoColor=white)](#install)
[![Model](https://img.shields.io/badge/Validated-Qwen3--4B--Instruct--2507-6F42C1)](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

![typing](https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=3200&pause=1000&color=58A6FF&center=true&vCenter=true&width=760&height=60&lines=Extract%20the%20Chinese%20AI-register%20direction%20from%20Qwen3%20residuals%3BProject%20it%20out%20at%20inference%3A%20x%20%E2%86%90%20x%20%E2%88%92%20%28x%C2%B7d%CC%82%29d%CC%82%3BParallel%20triples%2C%20stock%20markers%2C%20hype%20cadence%3BGone%20at%20the%20generation%20source%2C%20not%20in%20the%20edit%20layer)

```bash
uv run zhsteer demo --model qwen3-4b     # → assets/demo/ablation-report.md
```

Free + open source · MIT · no paid tier

</div>

## The problem it solves

Chinese content pipelines running Qwen/GLM/DeepSeek fight the models' baked-in AI register every day: parallel triples, the “值得注意的是” (worth noting) family, “随着……的不断发展” openers, hype cadence. The existing fixes are prompt-layer skills (the blader/humanizer lineage, Humanizer-zh) that rewrite each output at the edit layer. Both approaches treat symptoms — the register grows back on the next `generate()`, and every model version bump invalidates the rule library.

zhsteer moves the same job into the weights:

1. Feed Qwen3-4B 64 hand-written Chinese minimal pairs (the same content in AI register vs plain register) and take the difference-in-means over the residual stream. That yields a single "Chinese AI-register direction".
2. At generation time, project that direction out of the residual stream: `x ← x − (x·d̂)d̂`. The register never reaches the output, while the orthogonal content subspace is untouched.
3. It ships as a callable inference primitive — a `steer()` context manager or `--ablate 0.9` — that plugs next to your existing transformers call like `temperature` does. Extraction runs once per model version; an Agent content pipeline stops re-washing every article.

## Read this first: what v0.1 honestly claims

- **Validated on Qwen/Qwen3-4B-Instruct-2507 only.** DirectionPacks are model-tagged and checksum-verified; loading one into a different model is rejected.
- **Chinese-output separability is exactly the falsifier experiment v0.1 exists to run.** The plan's falsification bars: ≥ 90% held-out pair separability, ≥ 20% mean trope-score reduction, zero degenerate outputs. These numbers should come from your own GPU — run `zhsteer demo` and read the report. If the bars are not met, the negative result is worth publishing too; that is why the report, not the repo, is the launch artifact.
- **The bundled [`assets/demo/ablation-report.md`](./assets/demo/ablation-report.md) is a report template** — the full structure with placeholder values, not measured results. Your own `zhsteer demo` run overwrites that path with measured data.

## Install

Requirements: NVIDIA GPU (≥ 16GB VRAM), Python 3.12+, [uv](https://docs.astral.sh/uv/). macOS and CPU inference are out of scope for v0.1.

```bash
git clone https://github.com/SuperMarioYL/zhsteer
cd zhsteer
uv sync
uv run zhsteer doctor    # checks CUDA, VRAM, and disk for the model download
```

No CUDA or not enough VRAM? `doctor` lists exactly what is missing instead of letting you wait through an 8GB download for nothing.

## Quickstart

```bash
# One command: download the model (~8GB first run) → extract the direction →
# regenerate 10 held-out prompts → score both sides → write the report
# ~30 minutes expected on a 16GB consumer GPU; the model downloads once
uv run zhsteer demo --model qwen3-4b
```

Step by step, if you prefer:

```bash
uv run zhsteer extract --model qwen3-4b            # pairs/seed_pairs.zh.yaml → packs/qwen3-4b.pack
uv run zhsteer generate --pack packs/qwen3-4b.pack \
    --ablate 0.9 --prompt "写一段产品介绍"           # before/after side by side + trope scores
```

## Demo report

The output of `zhsteer demo` is an ablation report: full Before/After text for every prompt, deterministic trope scores, degenerate-output checks, and a summary table. **[See the report template bundled in this repo →](./assets/demo/ablation-report.md)** (placeholder structure; running the demo overwrites it with your measured data). It looks like this:

```markdown
# zhsteer ablation report
- Model: Qwen/Qwen3-4B-Instruct-2507
- Direction pack: packs/qwen3-4b.pack (layer <L>, checksum <…>)
- Minimal pairs: pairs/seed_pairs.zh.yaml (52 train / 12 held out)

## Before/after by prompt
### 1. <prompt summary>
**Before (plain)**                     **After (ablated, strength 1.0)**
<original output, register intact>     <output with the direction projected out>

trope score: <XX.XX> → <XX.XX> (<XX%> lower); degeneracy check: pass

## Score summary
| # | Prompt | Before | After | Reduction |
| 1 | <…>    | <XX.XX> | <XX.XX> | <XX%> |
Mean trope score: <XX.XX> → <XX.XX> (<XX%> mean reduction)
```

The scorer is a deterministic lexicon + regex implementation — the same text always produces the same score, so the before/after delta can only come from the ablation itself.

## How it works

```text
zhsteer CLI (typer)
 ├─ loader.py   — model loading: CUDA / bf16 (generation hooks live in ablate)
 ├─ extract.py  — minimal-pair activations (per-layer mean pooling) → diff-in-means direction
 ├─ ablate.py   — generation-time projection x ← x − (x·d̂)d̂, tunable strength
 ├─ score.py    — deterministic trope scorer (parallel triples / markers / hype lexicon)
 └─ pack.py     — DirectionPack save / load / verify (schema v1, checksummed)

pairs/seed_pairs.zh.yaml ──extract──▶ packs/qwen3-4b.pack ──steer()──▶ ablated output ──score──▶ ablation-report.md
```

The core data structure is the **DirectionPack**: a sha256-checksummed, model-tagged artifact carrying the direction vector plus the validation evidence gathered at extraction time (per-layer separability, train/held-out split, sample counts). On load, the checksum and every field are recomputed — a hand-edited or corrupted pack fails loudly instead of steering generation with a broken direction.

Layer convention: the direction is recorded "after decoder layer *i*" — the exact point where the ablation hook attaches, so the extracted layer transfers directly to ablation. The default layer is the one with the highest training-pair separability (deepest on ties); held-out separability is recorded in the pack as an honest generalization number and never participates in selection.

## Usage

**Plugging into your own pipeline** (3 core lines; full example in [`examples/steer_demo.py`](./examples/steer_demo.py)):

```python
from zhsteer import load, steer

pack = load("packs/qwen3-4b.pack")
with steer(model, pack, strength=0.9):
    output = model.generate(**inputs)   # your existing transformers call, unchanged
```

`strength=1.0` removes exactly the register component; lower values damp it, higher values over-remove. The hook is attached on context entry and removed on exit — even if the block raises, it cannot leak into surrounding generations.

**Commands**:

| Command | What it does |
| --- | --- |
| `zhsteer doctor` | Check CUDA, ≥ 16GB VRAM, and disk for the model download |
| `zhsteer extract --model qwen3-4b` | Extract the direction from the bundled 64 minimal pairs, write `packs/qwen3-4b.pack` |
| `zhsteer generate --pack <pack> --prompt <…> --ablate 0.9` | Generate one prompt twice, side by side, with scores |
| `zhsteer demo --model qwen3-4b` | Run the whole chain in one command and write the ablation report |
| `zhsteer score --text <…>` | Score a finished text with the deterministic trope scorer (no model needed) |

Have your own minimal pairs? Pass `--pairs your_pairs.yaml`; the schema matches the bundled file (`ai` / `plain` / `held_out`).

## Scoring contract

The scorer covers nine classes of Chinese AI-register signals: parallel triples, the “值得” marker family, summary boilerplate, empty connectives, progressive constructions, era openers, first/second enumerations, a hype lexicon, and exclamation density. The score is weighted hits per 1,000 non-whitespace characters — density keeps before/after comparable at different lengths. Empty or degenerate output is a quality failure handled by the separate degeneracy check (verbatim repetition detection), not by the register score.

## Known limitations

- v0.1 validates one model, Qwen3-4B-Instruct-2507; switching models requires re-extracting the pack (the schema is model-tagged, so mismatches fail loudly).
- HF transformers inference only; vLLM / SGLang / OpenAI-compatible serving integration is not implemented.
- Single direction, single layer; multi-direction composition (style × format × language) is out of scope for v0.1.
- The scorer is a deterministic lexicon metric, not an LLM-as-judge review — it measures register density, not writing quality.
- The bundled minimal pairs are hand-written and cover common content scenarios, not an exhaustive set; pass `--pairs` with your own set for custom domains.

## Roadmap

All of the following are **unimplemented** plans:

- [ ] GLM-4 / DeepSeek direction-pack validation (the schema is ready)
- [ ] vLLM / SGLang serving integration, turning `--ablate` into a one-line serving parameter
- [ ] Multi-direction composition (style × format × language)
- [ ] A pack release cadence: every model version bump ships fresh report numbers

## Changelog

See [CHANGELOG.md](./CHANGELOG.md).

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
