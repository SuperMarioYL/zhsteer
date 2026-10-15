"""Extract the Chinese AI-register direction from Qwen residual-stream activations.

Pipeline (milestone m1):

1. Read the bundled Chinese minimal pairs — each pair is the same content
   written twice, once in AI register (排比三连, "值得注意的是", 爆款体 cadence)
   and once plain.
2. Run both sides through the model and mean-pool the residual stream after
   every decoder layer. ``output_hidden_states=True`` yields all layers from a
   single forward per batch, so no per-layer hook registry is needed for
   extraction (the loader's hook registry is a runtime/ablation concern).
3. Per layer, take the unit-normalized difference of class means
   ``mean(AI) - mean(plain)``.
4. Select the layer by training-pair separability and record the honest
   held-out number in the pack's ``validation`` evidence. Separating >=90% of
   held-out pairs by projection sign is m1's done bar — and the plan's kill
   experiment, so the number is reported as measured, never gated.

Pair file schema (YAML; ``pairs/seed_pairs.zh.yaml``)::

    language: zh            # optional, informational
    pairs:
      - ai: "...AI 腔版本..."
        plain: "...同内容的朴素版本..."
        held_out: true      # optional; excluded from extraction, used for validation

Layer convention: layer *i* means the residual stream after decoder layer *i*
(``hidden_states[i+1]``) — the exact point a generation-time ablation hook on
decoder layer *i* would modify, so the recorded layer transfers directly to
ablation.

Sign convention: the direction points from plain toward AI register. Ablation
subtracts ``alpha * (x @ d) * d`` to remove the register component, and a pair
counts as separated when the AI side projects strictly higher than the plain
side.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from zhsteer.pack import PACK_SCHEMA, DirectionPack

DEFAULT_BATCH_SIZE = 8

_PAIR_KEYS = {"ai", "plain", "held_out"}


@dataclass(frozen=True)
class MinimalPair:
    ai: str
    plain: str
    held_out: bool = False


@dataclass(frozen=True)
class PairSet:
    pairs: tuple[MinimalPair, ...]
    source: str
    sha256: str  # over the raw file bytes: provenance of the exact input used


def load_pairs(path: str | Path) -> PairSet:
    """Read and validate the minimal-pair YAML file."""
    path = Path(path)
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    if not isinstance(doc, dict) or not isinstance(doc.get("pairs"), list) or not doc["pairs"]:
        raise ValueError(f"{path}: expected a mapping with a non-empty 'pairs' list")

    pairs: list[MinimalPair] = []
    for index, entry in enumerate(doc["pairs"]):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: pair #{index} must be a mapping")
        unknown = set(entry) - _PAIR_KEYS
        if unknown:
            # Strict on purpose: a misspelled "held_out" would otherwise
            # silently move pairs between the extraction and validation sets.
            raise ValueError(f"{path}: pair #{index} has unknown keys {sorted(unknown)}")
        texts = {}
        for key in ("ai", "plain"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: pair #{index} needs a non-empty {key!r} text")
            texts[key] = value
        held_out = entry.get("held_out", False)
        if not isinstance(held_out, bool):
            raise ValueError(f"{path}: pair #{index}: 'held_out' must be a boolean")
        pairs.append(MinimalPair(ai=texts["ai"], plain=texts["plain"], held_out=held_out))

    return PairSet(pairs=tuple(pairs), source=str(path), sha256=hashlib.sha256(raw).hexdigest())


def capture_activations(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> torch.Tensor:
    """Mean-pool the residual stream after every decoder layer.

    Returns a float32 CPU tensor ``[n_layers, n_texts, d_model]``: row *i* is
    the attention-masked token mean of the residual stream after decoder layer
    *i* for each text. Masked means make the result independent of batch
    padding, so activations captured in one pass can be sliced into arbitrary
    train/held-out subsets afterwards.
    """
    if not texts:
        raise ValueError("no texts to capture activations from")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        # Batched tokenization needs a pad token; the EOS token is the
        # conventional stand-in and is masked out of every mean below.
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    per_layer: list[list[torch.Tensor]] = []
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    list(texts[start : start + batch_size]),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = model(**encoded, output_hidden_states=True, use_cache=False)
                hidden_states = output.hidden_states  # (n_layers + 1) × [batch, seq, d_model]
                if not per_layer:
                    per_layer = [[] for _ in range(len(hidden_states) - 1)]
                mask = encoded["attention_mask"].to(hidden_states[0].dtype).unsqueeze(-1)
                for layer_index in range(1, len(hidden_states)):
                    residual = hidden_states[layer_index].to(torch.float32)
                    mean = (residual * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                    per_layer[layer_index - 1].append(mean.cpu())
    finally:
        if was_training:
            model.train()

    return torch.stack([torch.cat(chunks) for chunks in per_layer])


def diff_in_means(ai: torch.Tensor, plain: torch.Tensor) -> torch.Tensor:
    """Unit-norm ``mean(ai) - mean(plain)`` from stacked activations ``[n, d_model]``."""
    direction = ai.mean(dim=0) - plain.mean(dim=0)
    norm = direction.norm()
    if not torch.isfinite(norm) or norm.item() == 0.0:
        raise ValueError("degenerate activations: diff-in-means direction has zero norm")
    return (direction / norm).to(torch.float32)


def separability(direction: torch.Tensor, ai: torch.Tensor, plain: torch.Tensor) -> float:
    """Fraction of matched pairs whose AI side projects strictly above the plain side."""
    if ai.shape[0] == 0 or plain.shape[0] == 0:
        raise ValueError("separability needs at least one pair on each side")
    margin = (ai @ direction) - (plain @ direction)
    return (margin > 0).float().mean().item()


def extract_direction(
    model,
    tokenizer,
    pairs_path: str | Path,
    *,
    model_id: str | None = None,
    layer: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DirectionPack:
    """Extract the register direction and wrap it in a validated DirectionPack.

    ``layer=None`` picks the decoder layer with the highest training-pair
    separability (deepest on ties — later layers integrate more of the register
    computation before it reaches the unembed, and mid-network hook points
    leave earlier layers free to keep steering content). Selection deliberately
    uses the training split only; the held-out separability reported in
    ``validation`` stays an honest generalization number.
    """
    pairset = load_pairs(pairs_path)
    pairs = pairset.pairs
    train_index = [i for i, pair in enumerate(pairs) if not pair.held_out]
    held_out_index = [i for i, pair in enumerate(pairs) if pair.held_out]
    if not train_index:
        raise ValueError("every pair is marked held_out; nothing left to extract the direction from")
    split = bool(held_out_index)
    if not split:
        # No explicit split: the "held-out" numbers below are in-sample. The
        # held_out_split flag keeps that interpretable instead of misleading.
        held_out_index = list(range(len(pairs)))

    acts_ai = capture_activations(model, tokenizer, [p.ai for p in pairs], batch_size=batch_size)
    acts_plain = capture_activations(model, tokenizer, [p.plain for p in pairs], batch_size=batch_size)
    n_layers = acts_ai.shape[0]

    if layer is not None and not 0 <= layer < n_layers:
        raise ValueError(f"layer must be in [0, {n_layers}), got {layer}")
    layer_candidates = range(n_layers) if layer is None else [layer]

    train_ai, train_plain = acts_ai[:, train_index, :], acts_plain[:, train_index, :]
    held_ai, held_plain = acts_ai[:, held_out_index, :], acts_plain[:, held_out_index, :]

    directions: dict[int, torch.Tensor] = {}
    train_sep: dict[int, float] = {}
    held_sep: dict[int, float] = {}
    for index in layer_candidates:
        direction = diff_in_means(train_ai[index], train_plain[index])
        directions[index] = direction
        train_sep[index] = separability(direction, train_ai[index], train_plain[index])
        held_sep[index] = separability(direction, held_ai[index], held_plain[index])

    if layer is None:
        chosen = max(layer_candidates, key=lambda i: (train_sep[i], i))
        selection = "layer with the highest training-pair separability (deepest on ties)"
    else:
        chosen = layer
        selection = f"layer forced by caller (layer={layer})"

    if model_id is None:
        model_id = getattr(getattr(model, "config", None), "_name_or_path", None)
    if not model_id:
        raise ValueError("model_id is required: pass the HF repo id or load the model from the hub")

    return DirectionPack(
        schema=PACK_SCHEMA,
        model=model_id,
        layer=chosen,
        direction=directions[chosen],
        pairs_hash=pairset.sha256,
        validation={
            "pairs_file": pairset.source,
            "aggregation": "mean",
            "selection": selection,
            "held_out_split": split,
            "n_pairs": len(pairs),
            "n_train": len(train_index),
            "n_held_out": len(held_out_index),
            "n_layers": n_layers,
            "layer": chosen,
            "train_separability": train_sep[chosen],
            "held_out_separability": held_sep[chosen],
            "separability_by_layer": {
                str(i): {"train": train_sep[i], "held_out": held_sep[i]}
                for i in sorted(train_sep)
            },
        },
    )
