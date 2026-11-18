# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is semver.

## [0.2.0] - 2026-11-19

### Fixed

- `zhsteer demo` no longer reports a fully degenerate ablation as a success: the CLI summary prints the After-side degeneracy count and exits non-zero when every steered output is empty or collapsed into repetition (the report is still written first). Previously that run printed a 100% score reduction and exited 0.
- `zhsteer extract` no longer presents in-sample separability as held-out evidence when the pair file has no `held_out` split: the second number is labelled in-sample, the ≥90% held-out falsifier is stated as unevaluated, and the ablation report uses the same wording.
- `zhsteer.pack.load` raises `PackError` for garbage or truncated pack files instead of leaking torch's `UnpicklingError`/`RuntimeError`, so the CLI prints one line instead of a Python traceback.
- Masked mean-pooling sums the attention mask in float32. The previous bf16 sum under-counted tokens above 256 and biased every pooled activation (a 257-token text pooled to 1.00390625 instead of 1.0), which fed the extracted direction and the recorded separability.

### Changed

- The ablation report records extraction provenance from the pack's own pairs hash, lists the prompt file separately when it differs, and prints a repro command built from the flags actually used (`--pack`, `--pairs`, `--out`, `--strength`, `--prompts`, `--max-new-tokens`, `--batch-size`).
- `zhsteer --version` prints the package version.
- Version lockstep to 0.2.0 (`VERSION`, `pyproject.toml`, `__version__`) with 16 new tests (41 → 57), including CLI-level tests for the report and demo summary.

### Known limits

- Unchanged from 0.1.0: validated on Qwen/Qwen3-4B-Instruct-2507 only; CUDA >= 16GB VRAM; HF transformers hooks only; deterministic lexicon scorer.

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
