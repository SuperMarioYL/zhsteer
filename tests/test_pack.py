"""DirectionPack integrity tests: round-trip, checksum, and every verify rule."""

from __future__ import annotations

import pytest
import torch

from zhsteer.pack import (
    PACK_SCHEMA,
    DirectionPack,
    PackError,
    load,
    pack_checksum,
    save,
    verify,
)

D_MODEL = 16


def make_pack(**overrides) -> DirectionPack:
    generator = torch.Generator().manual_seed(overrides.pop("seed", 0))
    direction = torch.randn(D_MODEL, generator=generator)
    direction = (direction / direction.norm()).to(torch.float32)
    fields = dict(
        schema=PACK_SCHEMA,
        model="Qwen/Qwen3-4B-Instruct-2507",
        layer=21,
        direction=direction,
        pairs_hash="ab" * 32,
        validation={"train_separability": 0.98, "held_out_separability": 0.92},
    )
    fields.update(overrides)
    return DirectionPack(**fields)


# --------------------------------------------------------------------------- round trip


def test_save_and_load_round_trip(tmp_path):
    pack = make_pack()
    path = save(pack, tmp_path / "sub" / "qwen3-4b.pack")  # parent dirs created
    loaded = load(path)  # verifies the checksum by default

    assert loaded.schema == pack.schema
    assert loaded.model == pack.model
    assert loaded.layer == pack.layer
    assert loaded.pairs_hash == pack.pairs_hash
    assert loaded.validation == pack.validation
    assert torch.equal(loaded.direction, pack.direction)
    assert loaded.direction.dtype is torch.float32


def test_checksum_is_deterministic_and_field_sensitive(tmp_path):
    pack = make_pack()
    save(pack, tmp_path / "a.pack")
    save(make_pack(seed=0), tmp_path / "b.pack")
    assert pack_checksum(pack) == pack_checksum(make_pack(seed=0))

    shifted = make_pack(seed=1, layer=pack.layer + 1)
    assert pack_checksum(shifted) != pack_checksum(pack)


def test_tampered_direction_fails_checksum(tmp_path):
    path = save(make_pack(), tmp_path / "pack")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    # flip stays unit-norm and finite, so the structural checks pass and the
    # checksum comparison is the only thing that can catch this
    payload["direction"] = torch.flip(payload["direction"], dims=[0])
    torch.save(payload, path)
    with pytest.raises(PackError, match="checksum mismatch"):
        load(path)


def test_tampered_validation_fails_checksum(tmp_path):
    path = save(make_pack(), tmp_path / "pack")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["validation"]["train_separability"] = 0.42
    torch.save(payload, path)
    with pytest.raises(PackError, match="checksum mismatch"):
        load(path)


def test_load_without_verification_is_an_explicit_escape_hatch(tmp_path):
    path = save(make_pack(), tmp_path / "pack")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["layer"] = 3
    torch.save(payload, path)
    loaded = load(path, verify_checksum=False)
    assert loaded.layer == 3


def test_load_rejects_unrelated_payloads(tmp_path):
    empty = tmp_path / "empty"
    torch.save({}, empty)
    with pytest.raises(PackError, match="missing field"):
        load(empty)

    unrelated = tmp_path / "unrelated"
    torch.save([1, 2, 3], unrelated)
    with pytest.raises(PackError, match="not a zhsteer pack"):
        load(unrelated)


# --------------------------------------------------------------------------- verify rules


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema": "other.pack/v1"}, "unknown pack schema"),
        ({"model": ""}, "non-empty model id"),
        ({"layer": -1}, "non-negative int"),
        ({"layer": True}, "non-negative int"),  # bool is an int subclass — rejected on purpose
        ({"pairs_hash": "ab" * 31}, "sha256 hex digest"),
        ({"pairs_hash": "zz" * 32}, "sha256 hex digest"),
        ({"validation": "nope"}, "validation must be a dict"),
        ({"direction": torch.ones(D_MODEL, dtype=torch.float64)}, "must be float32"),
        ({"direction": torch.ones(1, D_MODEL)}, "1-D"),
        ({"direction": torch.zeros(D_MODEL)}, "unit-norm"),
        ({"direction": torch.tensor([float("nan")] + [0.0] * (D_MODEL - 1))}, "non-finite"),
    ],
)
def test_verify_rejects_invalid_packs(overrides, message):
    with pytest.raises(PackError, match=message):
        verify(make_pack(**overrides))


def test_verify_accepts_a_valid_pack_without_checksum():
    verify(make_pack())  # must not raise
