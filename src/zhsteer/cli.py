"""zhsteer CLI (typer) — the plan's §3 happy path as commands.

- ``zhsteer doctor``   verify CUDA, >= 16GB VRAM and disk before the ~8GB download
- ``zhsteer extract``  minimal pairs -> diff-in-means direction -> checksummed DirectionPack
- ``zhsteer generate`` one prompt, generated twice: plain and with the register direction
                       projected out (``--ablate`` is the strength knob)
- ``zhsteer demo``     the one-command path: extract -> regenerate 10 held-out register
                       prompts -> score both sides -> write the ablation report
- ``zhsteer score``    the deterministic trope scorer, usable without a model

Heavy imports (torch / transformers) happen inside the commands that need them,
so ``zhsteer score`` and ``--help`` stay fast on machines without a GPU.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="把中文 AI 腔方向从 Qwen3 权重里投影掉 — 推理时消融，面向中文内容管线。",
)

DEFAULT_PAIRS = "pairs/seed_pairs.zh.yaml"
DEFAULT_REPORT = "assets/demo/ablation-report.md"

# Command defaults live here once: the typer options, the repro-command builder,
# and its "is this flag non-default?" checks all read these constants.
DEFAULT_STRENGTH = 1.0
DEFAULT_PROMPTS = 10
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_BATCH_SIZE = 8

# plan §3: doctor must verify CUDA, >= 16GB VRAM and disk for the ~8GB download.
MIN_VRAM_GB = 16.0
MIN_DISK_GB = 12.0


def _print_version(value: bool) -> None:
    if not value:
        return
    from zhsteer import __version__

    typer.echo(__version__)
    raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option("--version", help="打印版本号并退出", callback=_print_version, is_eager=True),
    ] = False,
) -> None:
    """zhsteer CLI."""


@contextlib.contextmanager
def _errors():
    """Turn known failures into one-line CLI errors instead of tracebacks."""
    try:
        yield
    except typer.Exit:  # an intentional exit (click.Exit subclasses RuntimeError)
        raise
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        typer.secho(f"错误：{exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from exc


def _resolve_pairs(pairs: str | None) -> Path:
    """Explicit --pairs, else the bundled seed set from the repo checkout."""
    if pairs:
        path = Path(pairs)
        if not path.is_file():
            raise FileNotFoundError(f"pairs 文件不存在：{path}")
        return path
    candidates = [Path(DEFAULT_PAIRS), Path(__file__).resolve().parents[2] / DEFAULT_PAIRS]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"未找到内置最小对（{DEFAULT_PAIRS}），请用 --pairs 指定")


def _pack_path(model_ref: str) -> Path:
    """Default pack location: packs/qwen3-4b.pack (alias) or packs/org--name.pack."""
    from zhsteer.loader import MODEL_ALIASES

    slug = MODEL_ALIASES.get(model_ref, model_ref.replace("/", "--"))
    return Path("packs") / f"{slug}.pack"


def _load_pack(path: Path, model_id: str):
    from zhsteer.pack import load as load_pack

    pack = load_pack(path)  # verifies the checksum
    if pack.model != model_id:
        raise ValueError(
            f"方向包属于 {pack.model}，与 --model {model_id} 不匹配 — "
            "pack 按模型打标，请用当前模型重新 zhsteer extract"
        )
    return pack


def _load_model(model_ref: str):
    from zhsteer.loader import load_model_and_tokenizer, resolve_model_id

    model_id = resolve_model_id(model_ref)
    typer.secho(f"加载模型 {model_id}（首次运行会下载 ~8GB）…", fg=typer.colors.CYAN, err=True)
    loaded = load_model_and_tokenizer(model_id)
    return loaded, model_id


def _pack_checksum_short(pack) -> str:
    from zhsteer.pack import pack_checksum

    return pack_checksum(pack)[:12]


def _separability_text(validation: dict) -> str:
    """Render separability evidence without passing in-sample numbers off as held-out.

    ``extract_direction`` falls back to validating on the training pairs when the
    pair file has no ``held_out`` entries and records ``held_out_split: false``;
    the second number is then in-sample, so it must not be read as the plan's
    >= 90% held-out falsifier.
    """
    train = validation["train_separability"]
    second = validation["held_out_separability"]
    if validation.get("held_out_split", True):
        return (
            f"train {train:.0%} / held-out {second:.0%}"
            f"（{validation['n_train']} train / {validation['n_held_out']} held-out）"
        )
    return (
        f"train {train:.0%} / in-sample {second:.0%}"
        f"（{validation['n_train']} 条全部参与训练；pairs 文件没有 held-out 划分，"
        ">=90% held-out 证伪线未评估）"
    )


@dataclass(frozen=True)
class _RunSummary:
    """One demo run's headline numbers; the report and the CLI both print these."""

    total: int
    mean_before: float
    mean_after: float
    mean_reduction: float
    degenerate_after: int


def _summarize(results: list) -> _RunSummary:
    """Aggregate ``results`` tuples once so report and CLI can never disagree."""
    n = max(len(results), 1)
    mean_before = sum(item[1].total for item in results) / n
    mean_after = sum(item[2].total for item in results) / n
    return _RunSummary(
        total=len(results),
        mean_before=mean_before,
        mean_after=mean_after,
        mean_reduction=(mean_before - mean_after) / mean_before if mean_before > 0 else 0.0,
        degenerate_after=sum(1 for item in results if not item[5]),
    )


def _repro_command(
    *,
    model: str,
    pairs: str | None,
    pack_path: Path | None,
    out: Path,
    strength: float,
    prompts: int,
    max_new_tokens: int,
    batch_size: int,
) -> str:
    """The command that reproduces this run: every flag that differed from its default."""
    parts = ["uv run zhsteer demo", f"--model {shlex.quote(model)}"]
    if pairs:
        parts.append(f"--pairs {shlex.quote(str(pairs))}")
    if pack_path is not None:
        parts.append(f"--pack {shlex.quote(str(pack_path))}")
    if out != Path(DEFAULT_REPORT):
        parts.append(f"--out {shlex.quote(str(out))}")
    if strength != DEFAULT_STRENGTH:
        parts.append(f"--strength {strength}")
    if prompts != DEFAULT_PROMPTS:
        parts.append(f"--prompts {prompts}")
    if max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
        parts.append(f"--max-new-tokens {max_new_tokens}")
    if batch_size != DEFAULT_BATCH_SIZE:
        parts.append(f"--batch-size {batch_size}")
    return " ".join(parts)


# --------------------------------------------------------------------------- doctor


@app.command()
def doctor() -> None:
    """体检本机是否满足 v0.2 的运行条件：CUDA、>= 16GB 显存、模型下载磁盘空间。"""
    import torch

    rows: list[tuple[str, bool, str]] = []
    rows.append(("Python >= 3.12", sys.version_info >= (3, 12), sys.version.split()[0]))
    rows.append(("torch", True, torch.__version__))

    try:
        import transformers

        rows.append(("transformers", True, transformers.__version__))
    except ImportError as exc:  # pragma: no cover - declared dependency, defensive only
        rows.append(("transformers", False, f"导入失败：{exc}"))

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        rows.append(("CUDA", True, f"{props.name}"))
        rows.append((f"显存 >= {MIN_VRAM_GB:.0f}GB", vram_gb >= MIN_VRAM_GB, f"{vram_gb:.1f}GB"))
        rows.append(("bf16", torch.cuda.is_bf16_supported(), "支持" if torch.cuda.is_bf16_supported() else "不支持，将回退 fp16"))
    else:
        rows.append(("CUDA", False, "无可用 CUDA 设备"))
        rows.append((f"显存 >= {MIN_VRAM_GB:.0f}GB", False, "未检测到 GPU"))

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    free_gb = shutil.disk_usage(cache if cache.exists() else Path.home()).free / 1024**3
    rows.append((f"磁盘可用 >= {MIN_DISK_GB:.0f}GB（模型缓存 {cache}）", free_gb >= MIN_DISK_GB, f"{free_gb:.1f}GB"))

    ok = True
    typer.echo("zhsteer doctor")
    for name, passed, detail in rows:
        ok &= passed
        mark = typer.style("PASS", fg=typer.colors.GREEN) if passed else typer.style("FAIL", fg=typer.colors.RED)
        typer.echo(f"  [{mark}] {name}：{detail}")

    if ok:
        typer.echo("可以运行 zhsteer demo --model qwen3-4b")
    else:
        typer.secho("未通过：v0.2 只支持 CUDA >= 16GB VRAM（macOS / CPU 不在支持范围）。", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)


# --------------------------------------------------------------------------- extract


@app.command()
def extract(
    model: Annotated[str, typer.Option("--model", "-m", help="模型别名或 HF repo id（v0.2 只验证 qwen3-4b）")] = "qwen3-4b",
    pairs: Annotated[str | None, typer.Option("--pairs", help="最小对 YAML（默认用内置 pairs/seed_pairs.zh.yaml）")] = None,
    out: Annotated[str | None, typer.Option("--out", "-o", help="方向包输出路径")] = None,
    layer: Annotated[int | None, typer.Option("--layer", min=0, help="强制指定层（默认自动选训练集可分性最高层）")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1, help="激活捕获批大小")] = DEFAULT_BATCH_SIZE,
) -> None:
    """从残差流提取中文 AI 腔方向，保存为带校验和的 DirectionPack。"""
    with _errors():
        import torch

        from zhsteer.extract import extract_direction
        from zhsteer.pack import pack_checksum, save as save_pack

        pairs_path = _resolve_pairs(pairs)
        out_path = Path(out) if out else _pack_path(model)
        loaded, model_id = _load_model(model)

        typer.echo(f"提取方向：{pairs_path}（layer={'auto' if layer is None else layer}）")
        pack = extract_direction(
            loaded.model,
            loaded.tokenizer,
            pairs_path,
            model_id=model_id,
            layer=layer,
            batch_size=batch_size,
        )
        saved = save_pack(pack, out_path)

        validation = pack.validation
        typer.echo(f"方向包：{saved}（layer {pack.layer}，d_model {pack.direction.numel()}，checksum {pack_checksum(pack)[:12]}…）")
        typer.echo(f"可分性 {_separability_text(validation)}")
        if validation.get("held_out_split", True):
            typer.echo("证伪线为 held-out 可分性 >= 90%（mvp_plan §8）——以上为实测值，请自行对照。")
        else:
            typer.secho(
                "注意：pairs 文件没有 held-out 划分，第二项是样本内可分性；"
                ">=90% 的 held-out 证伪线未评估。",
                fg=typer.colors.YELLOW,
                err=True,
            )


# --------------------------------------------------------------------------- generate


@app.command()
def generate(
    model: Annotated[str, typer.Option("--model", "-m", help="模型别名或 HF repo id")] = "qwen3-4b",
    pack_path: Annotated[Path, typer.Option("--pack", "-p", help="DirectionPack 路径（zhsteer extract 的输出）")] = ...,
    prompt: Annotated[str, typer.Option("--prompt", help="生成提示词")] = ...,
    ablate: Annotated[float, typer.Option("--ablate", "-a", min=0.0, help="消融强度：1.0 完全投影掉，0 关闭")] = DEFAULT_STRENGTH,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1, help="每侧生成的最大新 token 数")] = DEFAULT_MAX_NEW_TOKENS,
    raw: Annotated[bool, typer.Option("--raw", help="不套 chat template，按 base 模型补全方式喂提示词")] = False,
) -> None:
    """同一提示词生成两遍：原始 vs 投影消融，前后对照输出。"""
    with _errors():
        from zhsteer.ablate import generate_pair, is_degenerate
        from zhsteer.score import reduction, score_text

        loaded, model_id = _load_model(model)
        pack = _load_pack(pack_path, model_id)

        typer.echo(f"提示词：{prompt}")
        typer.echo(f"生成中（layer {pack.layer}，强度 {ablate}，greedy，max_new_tokens={max_new_tokens}）…")
        pair = generate_pair(
            loaded.model,
            loaded.tokenizer,
            prompt,
            pack,
            strength=ablate,
            max_new_tokens=max_new_tokens,
            use_chat_template=not raw,
        )

        before_score = score_text(pair.before)
        after_score = score_text(pair.after)
        delta = reduction(before_score, after_score)

        typer.echo("\n—— Before（原始）" + "—" * 40)
        typer.echo(pair.before or "（空输出）")
        typer.echo("\n—— After（消融）" + "—" * 40)
        typer.echo(pair.after or "（空输出）")
        typer.echo(f"\ntrope 分数：{before_score.total:.2f} → {after_score.total:.2f}（降幅 {delta:.0%}）")
        typer.echo(f"退化检查：Before {'通过' if not is_degenerate(pair.before) else '未通过'} / After {'通过' if not is_degenerate(pair.after) else '未通过'}")


# --------------------------------------------------------------------------- demo


def _render_report(
    *,
    model_id: str,
    pack_path: Path,
    pack,
    pairs_path: Path,
    validation: dict,
    strength: float,
    max_new_tokens: int,
    results: list,
    repro: str,
) -> str:
    """Assemble the markdown ablation report the demo command writes."""
    prompt_hash = validation.get("prompt_pairs_sha256")
    lines = [
        "# zhsteer 消融报告",
        "",
        f"- 运行日期：{date.today().isoformat()}",
        f"- 模型：{model_id}",
        f"- 方向包：`{pack_path}`（layer {pack.layer}，checksum `{_pack_checksum_short(pack)}`）",
        f"- 最小对（提取）：`{validation.get('pairs_file', '?')}`（sha256 `{pack.pairs_hash[:12]}…`，"
        f"train {validation['n_train']} / held-out {validation['n_held_out']}）",
    ]
    if prompt_hash and prompt_hash != pack.pairs_hash:
        # The pack's own pairs hash is the extraction provenance; the prompt file
        # is a separate input and must not be presented as the extraction source.
        lines.append(
            f"- 提示词来源：`{validation.get('prompt_pairs_path', pairs_path)}`"
            f"（sha256 `{prompt_hash[:12]}…`，与提取用最小对不是同一文件）"
        )
    lines += [
        f"- 消融强度：{strength} · 提示词数：{len(results)} · 生成参数：greedy，max_new_tokens={max_new_tokens}",
        f"- 复现：`{repro}`",
        "",
        "> 分数为实测值，不做门槛过滤。计划设定的证伪线（mvp_plan §8）：held-out 可分性 >= 90%，"
        "平均 trope 分数降幅 >= 20%，且无退化输出。提示词取自最小对集中标记 held_out 的 AI 腔侧，"
        "未参与方向提取。",
        "",
        f"## 方向包验证（提取时记录）",
        "",
        f"- 层选择：{validation['selection']}",
        f"- 可分性：{_separability_text(validation)}",
        "",
        "## 各提示词前后对比",
        "",
    ]

    for index, item in enumerate(results, start=1):
        pair, before_score, after_score, delta, before_ok, after_ok = item
        label = pair.prompt if len(pair.prompt) <= 24 else pair.prompt[:24] + "…"
        lines += [
            f"### {index}. {label}",
            "",
            f"**Before（原始）**",
            "",
            "```text",
            pair.before or "（空输出）",
            "```",
            "",
            f"**After（消融，强度 {pair.strength}）**",
            "",
            "```text",
            pair.after or "（空输出）",
            "```",
            "",
            f"trope 分数：{before_score.total:.2f} → {after_score.total:.2f}（降幅 {delta:.0%}）；"
            f"退化检查：Before {'通过' if before_ok else '未通过'} / After {'通过' if after_ok else '未通过'}",
            "",
        ]

    rows = []
    for index, item in enumerate(results, start=1):
        pair, before_score, after_score, delta, _, _ = item
        label = pair.prompt if len(pair.prompt) <= 16 else pair.prompt[:16] + "…"
        rows.append(f"| {index} | {label} | {before_score.total:.2f} | {after_score.total:.2f} | {delta:.0%} |")

    summary = _summarize(results)
    lines += [
        "## 分数汇总",
        "",
        "| # | 提示词 | Before | After | 降幅 |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
        f"**平均 trope 分数**：{summary.mean_before:.2f} → {summary.mean_after:.2f}"
        f"（平均降幅 {summary.mean_reduction:.0%}）",
        "",
        f"**退化输出**：{summary.degenerate_after}/{summary.total}（After 侧含成段重复或空输出即计为未通过）",
        "",
    ]
    return "\n".join(lines)


@app.command()
def demo(
    model: Annotated[str, typer.Option("--model", "-m", help="模型别名或 HF repo id（v0.2 只验证 qwen3-4b）")] = "qwen3-4b",
    pairs: Annotated[str | None, typer.Option("--pairs", help="最小对 YAML（默认用内置集合）")] = None,
    pack_path: Annotated[Path | None, typer.Option("--pack", "-p", help="复用已有方向包（默认现场重新提取）")] = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="报告输出路径")] = Path(DEFAULT_REPORT),
    strength: Annotated[float, typer.Option("--strength", min=0.0, help="消融强度")] = DEFAULT_STRENGTH,
    prompts: Annotated[int, typer.Option("--prompts", min=1, help="取多少条 held-out 提示词（计划口径为 10）")] = DEFAULT_PROMPTS,
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1)] = DEFAULT_MAX_NEW_TOKENS,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = DEFAULT_BATCH_SIZE,
) -> None:
    """一条命令跑通：提取方向 -> 重新生成 held-out 提示词 -> 打分 -> 写消融报告。"""
    with _errors():
        from zhsteer.ablate import generate_pair, is_degenerate
        from zhsteer.extract import load_pairs
        from zhsteer.score import reduction, score_text

        reused_pack = pack_path is not None
        pairs_path = _resolve_pairs(pairs)
        pairset = load_pairs(pairs_path)
        held_out = [p for p in pairset.pairs if p.held_out]
        if not held_out:
            raise ValueError("最小对集中没有 held_out 对，无法构成验证提示词（内置集合带 12 条）")
        if len(held_out) < prompts:
            typer.secho(
                f"注意：held-out 对只有 {len(held_out)} 条，少于要求的 {prompts} 条，按实际数量执行。",
                fg=typer.colors.YELLOW,
                err=True,
            )
            prompts = len(held_out)
        selected = held_out[:prompts]

        loaded, model_id = _load_model(model)
        if pack_path is not None:
            pack = _load_pack(pack_path, model_id)
        else:
            from zhsteer.extract import extract_direction
            from zhsteer.pack import save as save_pack

            default_pack = _pack_path(model)
            typer.echo(f"提取方向并保存到 {default_pack} …")
            pack = extract_direction(
                loaded.model, loaded.tokenizer, pairs_path, model_id=model_id, batch_size=batch_size
            )
            save_pack(pack, default_pack)
            pack_path = default_pack

        validation = dict(pack.validation)
        # The pack's pairs_hash is the extraction provenance and is checksummed;
        # the prompt file is a separate input, recorded separately.
        validation["prompt_pairs_sha256"] = pairset.sha256
        validation["prompt_pairs_path"] = str(pairs_path)
        validation["selection"] = validation.get("selection", "")

        results = []
        for index, pair in enumerate(selected, start=1):
            typer.echo(f"[{index}/{len(selected)}] {pair.ai[:30]}…")
            generated = generate_pair(
                loaded.model,
                loaded.tokenizer,
                pair.ai,
                pack,
                strength=strength,
                max_new_tokens=max_new_tokens,
            )
            before_score = score_text(generated.before)
            after_score = score_text(generated.after)
            results.append(
                (
                    generated,
                    before_score,
                    after_score,
                    reduction(before_score, after_score),
                    not is_degenerate(generated.before),
                    not is_degenerate(generated.after),
                )
            )

        report = _render_report(
            model_id=model_id,
            pack_path=pack_path,
            pack=pack,
            pairs_path=pairs_path,
            validation=validation,
            strength=strength,
            max_new_tokens=max_new_tokens,
            results=results,
            repro=_repro_command(
                model=model,
                pairs=pairs,
                pack_path=pack_path if reused_pack else None,
                out=out,
                strength=strength,
                prompts=len(selected),
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
            ),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")

        summary = _summarize(results)
        typer.echo(f"报告已写入 {out}")
        typer.echo(
            f"平均 trope 分数：{summary.mean_before:.2f} → {summary.mean_after:.2f}"
            f"（平均降幅 {summary.mean_reduction:.0%}）"
        )
        typer.echo(
            f"退化输出：{summary.degenerate_after}/{summary.total}"
            "（After 侧；空输出或成段重复即未通过）"
        )
        if summary.total and summary.degenerate_after == summary.total:
            # Empty or collapsed output scores 0, which would otherwise read as a
            # perfect 100% reduction. A total collapse is a failed run, not a win.
            typer.secho(
                "错误：After 侧全部退化/为空 — 本次消融没有产出可用文本，不能视为成功；"
                "报告已保留供排查。",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)


# --------------------------------------------------------------------------- score


@app.command()
def score(
    text: Annotated[str | None, typer.Option("--text", "-t", help="要打分的文本")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="从文件读文本（默认读 stdin）")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="输出 JSON")] = False,
) -> None:
    """用确定性中文 AI 腔 trope 打分器给一段成品文本打分（不需要模型）。"""
    with _errors():
        from zhsteer.score import score_text

        if text is not None:
            content = text
        elif file is not None:
            content = file.read_text(encoding="utf-8")
        else:
            content = sys.stdin.read()
        if not content.strip():
            raise ValueError("没有输入文本：用 --text、--file 或 stdin 传入")

        result = score_text(content)
        if as_json:
            typer.echo(json.dumps({"total": result.total, "chars": result.chars, "hits": result.hits}, ensure_ascii=False))
            return

        typer.echo(f"trope 分数：{result.total:.2f}（每 1000 非空白字符的加权命中，非空白 {result.chars} 字符）")
        for trope in _tropes():
            count = result.hits[trope.name]
            if count:
                typer.echo(f"  {trope.label}（{trope.name}）：×{count}，权重 {trope.weight}")


def _tropes():
    from zhsteer.score import TROPES

    return TROPES


if __name__ == "__main__":
    app()
