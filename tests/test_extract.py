"""Extraction tests against a synthetic stand-in for the language model.

The fixture model never downloads anything: a tiny embedding table plus a
"register component" added at one chosen layer, so diff-in-means has a known
answer and the mechanical contract of ``extract`` (masked means, layer
selection, held-out bookkeeping, pack fields) can be asserted exactly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from zhsteer.extract import (
    capture_activations,
    diff_in_means,
    extract_direction,
    load_pairs,
    separability,
)
from zhsteer.pack import PACK_SCHEMA


# --------------------------------------------------------------------------- fixtures


class CharTokenizer:
    """Character-level tokenizer: id 0 is padding, real chars start at 1."""

    def __init__(self, texts):
        chars = sorted({c for text in texts for c in text})
        self.vocab = {c: i + 1 for i, c in enumerate(chars)}
        self.pad_token = None
        self.eos_token = "[EOS]"

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, **kwargs):
        rows = [[self.vocab.get(c, 0) for c in text] for text in texts]
        width = max(len(row) for row in rows)
        input_ids, attention_mask = [], []
        for row in rows:
            pad = [0] * (width - len(row))
            input_ids.append(row + pad)
            attention_mask.append([1] * len(row) + pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


class TinyLM(torch.nn.Module):
    """Embedding lookup + a register component injected at one layer's output.

    ``register_layer`` follows the extraction convention: hidden_states[i] is
    the residual stream after decoder layer i-1, so the component is added while
    producing hidden_states[register_layer + 1] — the exact tensor the
    extraction attributes to decoder layer ``register_layer``.
    """

    def __init__(self, vocab_size, marker_ids, d_model=8, n_layers=3, register_layer=2, strength=2.0, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn(vocab_size, d_model, generator=generator) * 0.05)
            vector = torch.randn(d_model, generator=generator)
        self.register_vector = vector / vector.norm()
        self.n_layers = n_layers
        self.register_layer = register_layer
        self.strength = strength
        self.marker_ids = torch.tensor(marker_ids)
        self.d_model = d_model

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **kwargs):
        hidden = self.embed(input_ids)
        states = [hidden]
        marker = torch.isin(input_ids, self.marker_ids).to(hidden.dtype).unsqueeze(-1)
        for index in range(self.n_layers):
            if index == self.register_layer:
                hidden = hidden + self.strength * marker * self.register_vector
            states.append(hidden)
        return type("Output", (), {"hidden_states": tuple(states)})()


def write_pairs(tmp_path, entries) -> str:
    path = tmp_path / "pairs.yaml"
    path.write_text(yaml.safe_dump({"language": "zh", "pairs": entries}, allow_unicode=True), encoding="utf-8")
    return str(path)


def default_entries(n_train=4, n_held=2):
    """Pairs whose AI sides carry two marker characters the plain sides never have."""
    filler = "甲乙丙丁戊己"
    entries = []
    for index in range(n_train + n_held):
        base = f"{filler[index]}句例文{index}内容"
        marker = "Ⓐ" if index % 2 == 0 else "Ⓑ"
        entries.append(
            {
                "ai": base + marker + marker,
                "plain": base,
                "held_out": index >= n_train,
            }
        )
    return entries


@pytest.fixture()
def tiny_setup(tmp_path):
    entries = default_entries()
    path = write_pairs(tmp_path, entries)
    alphabet = [e["ai"] + e["plain"] for e in entries] + ["[EOS]"]
    tokenizer = CharTokenizer(alphabet)
    marker_ids = [tokenizer.vocab["Ⓐ"], tokenizer.vocab["Ⓑ"]]
    model = TinyLM(len(tokenizer.vocab) + 2, marker_ids)
    return model, tokenizer, path


# --------------------------------------------------------------------------- load_pairs


def test_load_pairs_parses_and_hashes(tmp_path):
    entries = default_entries()
    path = write_pairs(tmp_path, entries)
    pairset = load_pairs(path)
    assert len(pairset.pairs) == len(entries)
    assert pairset.pairs[0].held_out is False
    assert pairset.pairs[-1].held_out is True
    assert pairset.sha256 == hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_load_pairs_rejects_misspelled_split_flag(tmp_path):
    entries = default_entries(1, 0)
    entries[0]["heldout"] = True  # a misspelling must not silently move pairs between splits
    with pytest.raises(ValueError, match="unknown keys"):
        load_pairs(write_pairs(tmp_path, entries))


def test_load_pairs_rejects_bad_entries(tmp_path):
    with pytest.raises(ValueError, match="non-empty 'pairs'"):
        load_pairs(write_pairs(tmp_path, []))
    with pytest.raises(ValueError, match="held_out' must be a boolean"):
        load_pairs(write_pairs(tmp_path, [{"ai": "x", "plain": "y", "held_out": "yes"}]))
    with pytest.raises(ValueError, match="non-empty 'plain'"):
        load_pairs(write_pairs(tmp_path, [{"ai": "x", "plain": "  "}]))


# --------------------------------------------------------------------------- math primitives


def test_diff_in_means_is_unit_and_points_to_ai_side():
    ai = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
    plain = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    direction = diff_in_means(ai, plain)
    assert direction.dtype is torch.float32
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_diff_in_means_rejects_degenerate_input():
    same = torch.ones(3, 4)
    with pytest.raises(ValueError, match="zero norm"):
        diff_in_means(same, same.clone())


def test_separability_counts_strict_sign_wins():
    direction = torch.tensor([1.0, 0.0])
    ai = torch.tensor([[2.0, 5.0], [3.0, -5.0]])
    plain = torch.tensor([[-1.0, 0.0], [-5.0, 0.0]])
    assert separability(direction, ai, plain) == 1.0
    assert separability(direction, plain, ai) == 0.0
    with pytest.raises(ValueError):
        separability(direction, torch.empty(0, 2), plain)


# --------------------------------------------------------------------------- capture


def test_capture_activations_is_padding_invariant(tiny_setup):
    model, tokenizer, _ = tiny_setup
    texts = ["甲句例文0内容", "乙句例文1内容ⒷⒷ", "丙句例文2内容"]
    one_by_one = capture_activations(model, tokenizer, texts, batch_size=1)
    one_batch = capture_activations(model, tokenizer, texts, batch_size=3)
    assert one_by_one.shape == (model.n_layers, len(texts), model.d_model)
    assert torch.allclose(one_by_one, one_batch, atol=1e-6)
    assert tokenizer.pad_token == tokenizer.eos_token  # batching engaged the fallback


def test_capture_activations_validates_input(tiny_setup):
    model, tokenizer, _ = tiny_setup
    with pytest.raises(ValueError, match="no texts"):
        capture_activations(model, tokenizer, [])
    with pytest.raises(ValueError, match="batch_size"):
        capture_activations(model, tokenizer, ["x"], batch_size=0)


# --------------------------------------------------------------------------- extract_direction


def test_extract_direction_finds_the_register_layer(tiny_setup):
    model, tokenizer, path = tiny_setup
    pack = extract_direction(model, tokenizer, path, model_id="Qwen/Qwen3-4B-Instruct-2507")

    assert pack.schema == PACK_SCHEMA
    assert pack.model == "Qwen/Qwen3-4B-Instruct-2507"
    assert pack.layer == 2  # the layer that actually carries the register component
    assert pack.validation["train_separability"] == 1.0
    assert pack.validation["held_out_separability"] == 1.0
    assert pack.validation["held_out_split"] is True
    assert (pack.validation["n_train"], pack.validation["n_held_out"]) == (4, 2)
    assert pack.direction.shape == (model.d_model,)
    assert abs(pack.direction.norm().item() - 1.0) < 1e-5


def test_extract_direction_without_split_marks_validation_in_sample(tiny_setup, tmp_path):
    model, tokenizer, _ = tiny_setup
    entries = [{"ai": e["ai"], "plain": e["plain"]} for e in default_entries()]
    path = write_pairs(tmp_path, entries)
    pack = extract_direction(model, tokenizer, path, model_id="m")
    assert pack.validation["held_out_split"] is False
    assert pack.validation["n_held_out"] == len(entries)
    assert pack.validation["train_separability"] == 1.0


def test_extract_direction_forced_layer_and_errors(tiny_setup):
    model, tokenizer, path = tiny_setup
    pack = extract_direction(model, tokenizer, path, model_id="m", layer=0)
    assert pack.layer == 0
    assert pack.validation["selection"].startswith("layer forced")

    with pytest.raises(ValueError, match="layer must be in"):
        extract_direction(model, tokenizer, path, model_id="m", layer=99)
    with pytest.raises(ValueError, match="model_id is required"):
        extract_direction(model, tokenizer, path)  # TinyLM exposes no .config to read


# --------------------------------------------------------------------------- bundled pairs contract


def test_bundled_seed_pairs_contract():
    """The README and plan both promise 64 hand-written pairs with 12 held out."""
    from zhsteer.score import score_text

    path = Path(__file__).resolve().parents[1] / "pairs" / "seed_pairs.zh.yaml"
    pairset = load_pairs(path)
    assert len(pairset.pairs) == 64
    assert sum(1 for pair in pairset.pairs if pair.held_out) == 12

    plain_total = ai_total = 0.0
    for pair in pairset.pairs:
        assert pair.ai.strip() and pair.plain.strip()
        assert pair.ai != pair.plain
        plain_total += score_text(pair.plain).total
        ai_total += score_text(pair.ai).total
    assert plain_total == 0.0  # plain sides carry no scorer-recognized register
    assert ai_total > 30 * len(pairset.pairs)  # ai sides are register-heavy on average
