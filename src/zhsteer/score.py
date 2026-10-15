"""Deterministic Chinese AI-register trope scorer.

A pure lexicon + regex scorer over the finished text: no model, no sampling,
no judge — the same text always produces the same :class:`TropeScore`. That is
what makes the demo report's before/after delta trustworthy: the only thing
that can move the number is the ablation.

Coverage follows the plan's spec — 排比三连, the "值得注意的是" family, and a
爆款体 lexicon — plus the surrounding stock-phrase frames Chinese AI register
leans on (总结套话, 空话连接, 递进句式, 万能开头, 首先其次体, 感叹号). Each trope
is one weighted regex; the 排比三连 pattern is a backreference regex over
comma-separated clauses. Weights are small so a single hit never dominates a
text, and every pattern is anchored to a recognizable stock phrase, so plain
prose stays near zero.

Scoring is density-based: weighted hits per 1000 non-whitespace characters.
Density keeps before/after comparable when the two sides generate different
lengths, which is what the plan's "排比密度" wording implies. Output in another
language or empty output simply scores ~0 on the Chinese lexicon — quality
failures of that kind are ``zhsteer.ablate.is_degenerate``'s job, not the
scorer's. :func:`reduction` turns two scores into the relative drop the demo
report and the m3 bar (>=20% on held-out register prompts) are stated in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Trope:
    """One weighted trope pattern; ``name`` is the stable key used in reports."""

    name: str
    label: str
    weight: float
    pattern: re.Pattern[str]


# 排比三连: three consecutive clauses (separated by ，,、；) sharing the same
# 1-2 leading CJK characters — "更高效、更智能、更普惠", "无论是…，无论是…，
# 无论是…". The shared prefix is restricted to CJK on purpose: repeated
# Latin/digit openings are lists ("iPhone 15、16、17"), not parallelism. The
# lookahead keeps overlapping triples counted, so a four-way 排比 scores as
# two triples and a three-way as one.
_CLAUSE = r"[^，,。；;！!？?\n\r]"
_SEP = r"[，,、；;]"
_PARALLEL_TRIPLE = re.compile(
    rf"(?=([\u4e00-\u9fff]{{1,2}}){_CLAUSE}{{1,20}}{_SEP}\1{_CLAUSE}{{1,20}}{_SEP}\1{_CLAUSE}{{1,20}})"
)

TROPES: tuple[Trope, ...] = (
    Trope("parallel_triple", "排比三连", 3.0, _PARALLEL_TRIPLE),
    Trope("discourse_marker", "值得体", 2.0, re.compile(
        "值得注意的是|值得一提|值得深思的是|值得警惕的是|值得关注的是|需要指出的是|需要注意的是")),
    Trope("summary_boilerplate", "总结套话", 2.0, re.compile(
        "综上所述|总而言之|总的来说|整体而言|综合来看|不难看出|不难发现|由此可见")),
    Trope("empty_connective", "空话连接", 1.0, re.compile(
        "与此同时|众所周知|不可否认|毋庸置疑|无独有偶|一言以蔽之")),
    Trope("progressive_pair", "递进句式", 1.0, re.compile(
        r"不仅[^。！？!?；;\n\r]{1,24}?(?:而且|还|更|也)")),
    Trope("era_opener", "万能开头", 1.0, re.compile(
        r"随着[^。！？!?，,；;\n\r]{0,20}的(?:不断|快速|迅猛|日益|飞速)?(?:发展|普及|进步|演进|增长|提升|流行|深入)"
        r"|在[^。！？!?，,；;\n\r]{1,16}的(?:今天|时代|当下)")),
    Trope("enumeration", "首先其次体", 1.0, re.compile(
        r"首先[^。！？!?；;\n\r]{1,60}?其次")),
    Trope("hype_lexicon", "爆款体", 2.0, re.compile(
        "家人们|宝子们|姐妹们|谁懂|绝绝子|yyds|天花板|炸裂|王炸|杀疯了|太顶了|拉满|拿捏"
        "|狠狠(?:拿捏|心动|种草|爱住|共情)|闭眼入|无限回购|史诗级|封神|绝了|无敌了|千万别|必看|码住"
        "|无脑冲|冲就完事|蹲一个|种草|拔草|避雷|踩雷|真香|上头|破防|超绝|绝美",
        re.IGNORECASE)),
    Trope("exclamation", "感叹号", 0.5, re.compile(r"[!！]")),
)


@dataclass(frozen=True)
class TropeScore:
    """Result of scoring one text.

    ``total`` is the headline number (weighted hits per 1000 non-whitespace
    characters); ``hits`` carries the raw per-trope counts, zero rows included,
    so a report can show which tropes the ablation actually removed.
    """

    total: float
    chars: int
    hits: dict[str, int]


def score_text(text: str) -> TropeScore:
    """Score one finished text; a pure function of the string."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    hits: dict[str, int] = {}
    weighted = 0.0
    for trope in TROPES:
        count = sum(1 for _ in trope.pattern.finditer(text))
        hits[trope.name] = count
        weighted += trope.weight * count
    chars = sum(1 for ch in text if not ch.isspace())
    return TropeScore(total=weighted * 1000.0 / max(chars, 1), chars=chars, hits=hits)


def reduction(before: TropeScore, after: TropeScore) -> float:
    """Relative trope-score drop from ``before`` to ``after`` (0.2 = 20% lower).

    Returns 0.0 when ``before`` carries no trope weight — nothing to measure —
    and goes negative if ablation made the text more trope-heavy; callers show
    the number as measured, never gated.
    """
    if before.total <= 0.0:
        return 0.0
    return (before.total - after.total) / before.total
