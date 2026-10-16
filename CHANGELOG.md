# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is semver.

## [0.1.0] - 2026-10-16

### Added

- Bundled seed set of 64 hand-written Chinese minimal pairs (AI register vs plain register), 12 of them held out from extraction for validation and demo prompts.
- `zhsteer extract`: per-layer residual-stream capture, diff-in-means direction extraction, and a checksummed, model-tagged DirectionPack artifact (`zhsteer.pack/v1`).
- `zhsteer generate`: generation-time projection-out hook with a strength knob (`--ablate`), printing plain and ablated outputs side by side with trope scores.
- `zhsteer demo`: the one-command path — extract, regenerate 10 held-out register prompts, score both sides, and write `assets/demo/ablation-report.md`.
- `zhsteer score`: the deterministic Chinese AI-register trope scorer (parallel triples, the “值得” marker family, summary boilerplate, hype lexicon, exclamation density), usable without a model.
- `zhsteer doctor`: pre-flight check for CUDA, >= 16GB VRAM, and disk space for the model download.
- Test suite covering extraction mechanics, pack integrity, and the trope scorer with synthetic fixtures (no model download required).

### Known limits

- Validated on Qwen/Qwen3-4B-Instruct-2507 only; packs are model-tagged and refuse mismatches.
- CUDA >= 16GB VRAM required; macOS and CPU inference are out of scope for v0.1.
