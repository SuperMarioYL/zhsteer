"""CLI-level tests: the report and the demo summary must describe what actually ran.

These cover the seams that reading the shipped code found wrong: a fully
degenerate ablation reported as a 100% success, in-sample separability presented
as held-out evidence, and a report whose provenance/repro line belongs to a
different run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from zhsteer import ablate, loader
from zhsteer.cli import _render_report, _repro_command, _separability_text, app
from zhsteer.extract import load_pairs
from zhsteer.pack import PACK_SCHEMA, DirectionPack, load, save

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REPORT = Path("assets/demo/ablation-report.md")
PAIRS_PATH = Path(__file__).resolve().parents[1] / "pairs" / "seed_pairs.zh.yaml"


# --------------------------------------------------------------------------- fixtures


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._parameter = torch.nn.Parameter(torch.zeros(1))


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    chat_template = None


def _make_pack(*, split: bool = True, n_train: int = 4, n_held_out: int = 2) -> DirectionPack:
    direction = torch.ones(4, dtype=torch.float32)
    direction = direction / direction.norm()
    return DirectionPack(
        schema=PACK_SCHEMA,
        model=MODEL_ID,
        layer=0,
        direction=direction,
        pairs_hash="ab" * 32,
        validation={
            "pairs_file": "/tmp/extract-pairs.yaml",
            "selection": "layer 0",
            "held_out_split": split,
            "n_train": n_train,
            "n_held_out": n_held_out,
            "train_separability": 1.0,
            "held_out_separability": 1.0,
        },
    )


@pytest.fixture()
def fake_model(monkeypatch):
    monkeypatch.setattr(
        loader,
        "load_model_and_tokenizer",
        lambda model_id: loader.Loaded(
            model=_FakeModel(), tokenizer=_FakeTokenizer(), model_id=model_id
        ),
    )


def _stub_generation(monkeypatch, *, before: str, after: str) -> None:
    def fake(model, tokenizer, prompt, pack, *, strength=1.0, **kwargs):
        return ablate.GenerationPair(
            prompt=prompt, before=before, after=after, layer=pack.layer, strength=strength
        )

    monkeypatch.setattr(ablate, "generate_pair", fake)


# --------------------------------------------------------------------------- demo summary


def test_demo_all_degenerate_exits_nonzero(tmp_path, fake_model, monkeypatch):
    pack = save(_make_pack(), tmp_path / "p2.pack")
    _stub_generation(monkeypatch, before="值得注意的是，更高效、更智能、更普惠！", after="")
    report = tmp_path / "report.md"

    result = CliRunner().invoke(
        app,
        ["demo", "--pack", str(pack), "--out", str(report), "--prompts", "2"],
    )

    assert result.exit_code == 1
    assert "退化输出：2/2" in result.output
    assert "全部退化" in result.output
    assert report.is_file()


def test_demo_healthy_run_exits_zero(tmp_path, fake_model, monkeypatch):
    pack = save(_make_pack(), tmp_path / "p2.pack")
    _stub_generation(
        monkeypatch,
        before="值得注意的是，更高效、更智能、更普惠！",
        after="这款耳机佩戴轻，续航标称三十小时，延迟实测八十毫秒左右，通勤开会都够用。",
    )
    report = tmp_path / "report.md"

    result = CliRunner().invoke(
        app,
        ["demo", "--pack", str(pack), "--out", str(report), "--prompts", "2"],
    )

    assert result.exit_code == 0, result.output
    assert "退化输出：0/2" in result.output
    assert report.is_file()


# --------------------------------------------------------------------------- separability label


def test_separability_text_labels_in_sample_without_split():
    validation = {
        "train_separability": 1.0,
        "held_out_separability": 1.0,
        "n_train": 6,
        "n_held_out": 6,
        "held_out_split": False,
    }
    text = _separability_text(validation)
    assert "in-sample" in text
    assert "未评估" in text
    assert "held-out 100%" not in text


def test_separability_text_keeps_held_out_wording_with_split():
    validation = {
        "train_separability": 1.0,
        "held_out_separability": 0.9,
        "n_train": 52,
        "n_held_out": 12,
        "held_out_split": True,
    }
    text = _separability_text(validation)
    assert "held-out 90%" in text
    assert "52 train / 12 held-out" in text
    assert "in-sample" not in text


def test_extract_command_reports_in_sample_when_pairs_have_no_split(
    tmp_path, fake_model, monkeypatch
):
    import zhsteer.extract as extract_module

    pack = _make_pack(split=False, n_train=6, n_held_out=6)
    monkeypatch.setattr(extract_module, "extract_direction", lambda *a, **k: pack)

    result = CliRunner().invoke(
        app, ["extract", "--pairs", str(PAIRS_PATH), "--out", str(tmp_path / "out.pack")]
    )

    assert result.exit_code == 0, result.output
    assert "in-sample" in result.output
    assert "证伪线未评估" in result.output
    assert "证伪线为 held-out" not in result.output


def test_extract_command_keeps_held_out_wording_with_split(tmp_path, fake_model, monkeypatch):
    import zhsteer.extract as extract_module

    pack = _make_pack(split=True, n_train=52, n_held_out=12)
    monkeypatch.setattr(extract_module, "extract_direction", lambda *a, **k: pack)

    result = CliRunner().invoke(
        app, ["extract", "--pairs", str(PAIRS_PATH), "--out", str(tmp_path / "out.pack")]
    )

    assert result.exit_code == 0, result.output
    assert "held-out 100%" in result.output
    assert "证伪线为 held-out 可分性" in result.output


# --------------------------------------------------------------------------- report provenance


def test_report_labels_pack_and_prompt_provenance(tmp_path):
    saved = save(_make_pack(), tmp_path / "p.pack")
    pack = load(saved)
    validation = dict(pack.validation)
    validation["prompt_pairs_sha256"] = "cd" * 32
    validation["prompt_pairs_path"] = "/tmp/prompt-pairs.yaml"

    report = _render_report(
        model_id=MODEL_ID,
        pack_path=saved,
        pack=pack,
        pairs_path=Path("/tmp/prompt-pairs.yaml"),
        validation=validation,
        strength=1.0,
        max_new_tokens=512,
        results=[],
        repro="uv run zhsteer demo --model qwen3-4b",
    )

    assert "最小对（提取）" in report
    assert "abababababab" in report
    assert "提示词来源" in report
    assert "cdcdcdcdcdcd" in report
    assert "不是同一文件" in report


def test_report_omits_prompt_source_when_hashes_match(tmp_path):
    saved = save(_make_pack(), tmp_path / "p.pack")
    pack = load(saved)
    validation = dict(pack.validation)
    validation["prompt_pairs_sha256"] = pack.pairs_hash

    report = _render_report(
        model_id=MODEL_ID,
        pack_path=saved,
        pack=pack,
        pairs_path=Path("/tmp/same-pairs.yaml"),
        validation=validation,
        strength=1.0,
        max_new_tokens=512,
        results=[],
        repro="uv run zhsteer demo --model qwen3-4b",
    )

    assert "最小对（提取）" in report
    assert "提示词来源" not in report


# --------------------------------------------------------------------------- repro command


def test_repro_command_is_short_for_a_default_run():
    command = _repro_command(
        model="qwen3-4b",
        pairs=None,
        pack_path=None,
        out=DEFAULT_REPORT,
        strength=1.0,
        prompts=10,
        max_new_tokens=512,
        batch_size=8,
    )
    assert command == "uv run zhsteer demo --model qwen3-4b"


def test_repro_command_includes_every_nondefault_flag():
    command = _repro_command(
        model="qwen3-4b",
        pairs="my pairs.yaml",
        pack_path=Path("packs/qwen3-4b.pack"),
        out=Path("reports/report.md"),
        strength=0.9,
        prompts=12,
        max_new_tokens=256,
        batch_size=4,
    )
    for fragment in (
        "--pairs 'my pairs.yaml'",
        "--pack packs/qwen3-4b.pack",
        "--out reports/report.md",
        "--strength 0.9",
        "--prompts 12",
        "--max-new-tokens 256",
        "--batch-size 4",
    ):
        assert fragment in command


# --------------------------------------------------------------------------- version


def test_cli_version_prints_the_package_version():
    from zhsteer import __version__

    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == __version__ == "0.2.0"


# --------------------------------------------------------------------------- bundled pairs sanity


def test_default_pairs_still_have_a_real_held_out_split():
    pairset = load_pairs(PAIRS_PATH)
    assert sum(1 for pair in pairset.pairs if pair.held_out) == 12
