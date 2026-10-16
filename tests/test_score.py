"""Trope scorer tests: the scorer is a pure function, so expectations are exact."""

from __future__ import annotations

import pytest

from zhsteer.score import TROPES, TropeScore, reduction, score_text


PLAIN = "这款耳机佩戴轻，续航标称三十小时，延迟实测八十毫秒左右，通勤开会都够用。"


def test_plain_prose_scores_zero():
    result = score_text(PLAIN)
    assert result.total == 0.0
    assert all(count == 0 for count in result.hits.values())


def test_every_trope_family_is_detected():
    cases = {
        "parallel_triple": "我们要做更高效、更智能、更普惠的工具",
        "discourse_marker": "值得注意的是，这个接口并不稳定",
        "summary_boilerplate": "综上所述，这款产品值得入手",
        "empty_connective": "与此同时，价格也降了下来",
        "progressive_pair": "它不仅省电，而且安静",
        "era_opener": "随着移动支付的不断发展，现金越来越少见了",
        "enumeration": "首先要校准，其次要试跑一遍",
        "hype_lexicon": "家人们谁懂，这支口红绝绝子",
        "exclamation": "太棒了！",
    }
    for name, text in cases.items():
        result = score_text(text)
        assert result.hits[name] >= 1, f"{name} was not detected in {text!r}"


def test_hype_lexicon_is_case_insensitive():
    assert score_text("这件衣服真的YYDS").hits["hype_lexicon"] >= 1


def test_parallel_triple_counts_overlapping_runs():
    # a four-way 排比 contains two overlapping triples
    result = score_text("更高效、更智能、更普惠、更安全")
    assert result.hits["parallel_triple"] == 2
    assert score_text("更高效、更智能、更普惠").hits["parallel_triple"] == 1


def test_repeated_latin_or_digit_openings_are_not_parallelism():
    # iPhone 15、16、17 is a list, not a 排比三连
    assert score_text("iPhone 15、16、17 都参加了以旧换新").hits["parallel_triple"] == 0


def test_density_is_length_invariant():
    once = score_text(PLAIN + "更高效、更智能、更普惠！")
    twice = score_text((PLAIN + "更高效、更智能、更普惠！") * 2)
    assert once.total == pytest.approx(twice.total)
    assert twice.chars == 2 * once.chars


def test_chars_counts_non_whitespace_only():
    result = score_text("a b\tc\nd")
    assert result.chars == 4


def test_hits_cover_every_trope_and_scores_are_deterministic():
    result = score_text("值得注意的是！")
    assert set(result.hits) == {trope.name for trope in TROPES}
    assert result == score_text("值得注意的是！")
    assert isinstance(result, TropeScore)


def test_empty_and_whitespace_text_score_zero_without_crashing():
    for text in ("", "   \n\t "):
        result = score_text(text)
        assert result.total == 0.0
        assert result.chars == 0


def test_score_text_rejects_non_string():
    with pytest.raises(TypeError, match="str"):
        score_text(123)  # type: ignore[arg-type]


def test_reduction_math():
    before = score_text("更高效、更智能、更普惠！值得注意的是！")
    after = score_text("更高效、更智能、更普惠！")
    drop = reduction(before, after)
    assert 0.0 < drop < 1.0
    assert drop == pytest.approx((before.total - after.total) / before.total)
    assert reduction(after, before) < 0  # getting worse reads as a negative drop
    empty = score_text("毫无套路的句子")
    assert reduction(empty, empty) == 0.0  # nothing to measure
