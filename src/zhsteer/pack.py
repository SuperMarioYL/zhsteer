"""DirectionPack — the checksummed, model-tagged register-direction artifact.

One pack carries a single diff-in-means direction for one model at one
residual-stream layer, plus the provenance (pairs hash) and the validation
evidence gathered at extraction time. ``save``/``load``/``verify`` are the
contract every other stage consumes: ``extract`` produces packs, inference-time
ablation loads them.

Integrity model: the file is a single ``torch.save`` payload whose ``checksum``
field is the sha256 of a canonical serialization of every semantic field.
``load`` recomputes the checksum from the loaded fields, so a hand-edited or
truncated file fails loudly instead of steering generation with a corrupted
direction. The canonical form (float32 little-endian direction bytes, JSON with
sorted keys) makes the checksum independent of torch's serialization details.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

PACK_SCHEMA = "zhsteer.pack/v1"

# ||direction|| == 1 by construction; float32 norm over d_model elements is
# accurate to ~1e-6, so 1e-4 leaves room without letting real drift through.
_NORM_ATOL = 1e-4


class PackError(ValueError):
    """A DirectionPack is structurally invalid, or fails its checksum."""


@dataclass
class DirectionPack:
    schema: str                # "zhsteer.pack/v1"
    model: str                 # e.g. "Qwen/Qwen3-4B-Instruct-2507"
    layer: int                 # residual-stream decoder layer the direction lives at
    direction: torch.Tensor    # [d_model], float32, unit-norm, diff-in-means
    pairs_hash: str            # sha256 of the minimal-pair file used
    validation: dict           # separability evidence (+ trope-delta once m3 adds it)


def _direction_bytes(direction: torch.Tensor) -> bytes:
    """Canonical byte form of the direction: float32, little-endian, CPU."""
    flat = direction.detach().cpu().to(torch.float32).reshape(-1)
    return flat.numpy().astype("<f4", copy=False).tobytes()


def _canonical_validation(validation: dict) -> bytes:
    return json.dumps(validation, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def pack_checksum(pack: DirectionPack) -> str:
    """sha256 over every semantic field; the checksum never covers itself."""
    digest = hashlib.sha256()
    digest.update(pack.schema.encode("utf-8"))
    digest.update(b"\0")
    digest.update(pack.model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(int(pack.layer)).encode("ascii"))
    digest.update(b"\0")
    digest.update(pack.pairs_hash.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_direction_bytes(pack.direction))
    digest.update(_canonical_validation(pack.validation))
    return digest.hexdigest()


def verify(pack: DirectionPack, *, checksum: str | None = None) -> None:
    """Validate a pack in memory; raise :class:`PackError` on any violation.

    With ``checksum`` set, also compare it against the recomputed checksum —
    the integrity check ``load`` applies.
    """
    if pack.schema != PACK_SCHEMA:
        raise PackError(f"unknown pack schema {pack.schema!r}, expected {PACK_SCHEMA!r}")
    if not isinstance(pack.model, str) or not pack.model:
        raise PackError("pack.model must be a non-empty model id")
    if isinstance(pack.layer, bool) or not isinstance(pack.layer, int) or pack.layer < 0:
        raise PackError(f"pack.layer must be a non-negative int, got {pack.layer!r}")
    if not isinstance(pack.pairs_hash, str) or len(pack.pairs_hash) != 64:
        raise PackError("pack.pairs_hash must be a sha256 hex digest")
    try:
        int(pack.pairs_hash, 16)
    except ValueError as exc:
        raise PackError("pack.pairs_hash must be a sha256 hex digest") from exc
    if not isinstance(pack.validation, dict):
        raise PackError("pack.validation must be a dict")

    direction = pack.direction
    if not isinstance(direction, torch.Tensor):
        raise PackError("pack.direction must be a torch.Tensor")
    if direction.dtype is not torch.float32:
        raise PackError(f"pack.direction must be float32, got {direction.dtype}")
    if direction.dim() != 1 or direction.numel() == 0:
        raise PackError(f"pack.direction must be 1-D [d_model], got shape {tuple(direction.shape)}")
    if not torch.isfinite(direction).all():
        raise PackError("pack.direction contains non-finite values")
    if abs(direction.norm().item() - 1.0) > _NORM_ATOL:
        raise PackError("pack.direction must be unit-norm")

    if checksum is not None and pack_checksum(pack) != checksum:
        raise PackError("checksum mismatch: the pack file was corrupted or edited")


def save(pack: DirectionPack, path: str | Path) -> Path:
    """Write the pack as a single checksummed artifact; returns the path."""
    verify(pack)
    payload = {
        "schema": pack.schema,
        "model": pack.model,
        "layer": int(pack.layer),
        "direction": pack.direction.detach().cpu().to(torch.float32).contiguous(),
        "pairs_hash": pack.pairs_hash,
        "validation": pack.validation,
        "checksum": pack_checksum(pack),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load(path: str | Path, *, verify_checksum: bool = True) -> DirectionPack:
    """Load and validate a pack written by :func:`save`."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise PackError(f"{path}: not a zhsteer pack file")
    try:
        pack = DirectionPack(
            schema=payload["schema"],
            model=payload["model"],
            layer=payload["layer"],
            direction=payload["direction"],
            pairs_hash=payload["pairs_hash"],
            validation=payload["validation"],
        )
        stored_checksum = payload["checksum"]
    except KeyError as exc:
        raise PackError(f"{path}: missing field {exc.args[0]!r}") from exc
    verify(pack, checksum=stored_checksum if verify_checksum else None)
    return pack
