"""zhsteer — 从 Qwen3 残差流提取中文 AI 腔方向，推理时投影消融。

公开 API（按需加载，避免 CLI 只想跑打分器时引入 torch）::

    from zhsteer import DirectionPack, load   # zhsteer.pack
    from zhsteer import steer, generate_pair  # zhsteer.ablate
    from zhsteer import score_text, reduction # zhsteer.score
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__version__ = "0.1.0"

try:  # installed via pip/uv — pyproject.toml is the single version source
    __version__ = version("zhsteer")
except PackageNotFoundError:  # running from an uninstalled checkout
    pass

_PACK_NAMES = ("PACK_SCHEMA", "PackError", "DirectionPack", "pack_checksum", "verify", "save", "load")
_ABLATE_NAMES = ("project_out", "steer", "GenerationPair", "generate_pair", "is_degenerate")
_SCORE_NAMES = ("Trope", "TROPES", "TropeScore", "score_text", "reduction")


def __getattr__(name: str):
    if name in _PACK_NAMES:
        from zhsteer import pack

        return getattr(pack, name)
    if name in _ABLATE_NAMES:
        from zhsteer import ablate

        return getattr(ablate, name)
    if name in _SCORE_NAMES:
        from zhsteer import score

        return getattr(score, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(["__version__", *_PACK_NAMES, *_ABLATE_NAMES, *_SCORE_NAMES])
