import gzip
import json

import pandas as pd

from scripts.check_missingness import (
    _has_desc,
    _image_urls,
    _mnar_verdict,
    _point_biserial,
    compute_stats,
    iter_jsonl,
    k_core_sets,
    parse_meta,
    parse_review,
    pattern_label,
)

# ---------------------------------------------------------------- 模态解析


def test_image_urls_variants():
    # list[dict]
    assert _image_urls([{"small": "a.jpg", "large": "b.jpg"}]) == ["a.jpg", "b.jpg"]
    # dict[list]（旧格式）
    assert _image_urls({"small": ["a.jpg"], "large": ["b.jpg"]}) == ["a.jpg", "b.jpg"]
    # list[str]
    assert _image_urls(["a.jpg", "b.jpg"]) == ["a.jpg", "b.jpg"]
    # 空 / 缺省 / 无意义
    assert _image_urls([]) == []
    assert _image_urls(None) == []
    assert _image_urls({"small": []}) == []
    # 空白 URL 忽略
    assert _image_urls([{"small": "  "}, "ok.jpg"]) == ["ok.jpg"]


def test_has_desc_string_and_list():
    assert _has_desc("good") is True
    assert _has_desc("   ") is False
    assert _has_desc(["line1", "line2"]) is True
    assert _has_desc(["  ", ""]) is False
    assert _has_desc([]) is False
    assert _has_desc(None) is False


def test_parse_meta_and_review():
    meta = parse_meta({"parent_asin": "A1", "title": "T", "description": "D", "images": [{"large": "x.jpg"}]})
    assert meta == {"has_title": True, "has_desc": True, "has_image": True, "n_images": 1}
    assert parse_meta({"parent_asin": "A2"}) == {
        "has_title": False, "has_desc": False, "has_image": False, "n_images": 0,
    }
    assert parse_review({"text": "nice"}) == {"has_text": True}
    assert parse_review({"text": ""}) == {"has_text": False}
    assert parse_review({"text": "   "}) == {"has_text": False}


def test_pattern_label():
    assert pattern_label(True, True, True) == "text+image+desc"
    assert pattern_label(True, True) == "text+image"
    assert pattern_label(True, False, True) == "text+desc"
    assert pattern_label(False, True, True) == "image+desc"
    assert pattern_label(True, False) == "text"
    assert pattern_label(False, True) == "image"
    assert pattern_label(False, False, True) == "desc"
    assert pattern_label(False, False) == "none"


# ---------------------------------------------------------------- k-core 一致性


def test_k_core_sets_iterative():
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u2", "u3", "u3"],
        "parent_asin": ["A", "B", "D", "A", "B", "D", "B", "C"],
        "timestamp": range(8),
    })
    # k=2 的级联剪枝：
    #   iter1: C 只有 u3 1 条 -> 剪；u3 本有 B+C 2 条存活
    #   iter2: 去掉 C 后 u3 只剩 B 1 条 -> 剪 u3
    #   iter3: 去掉 u3 的 B 后 B 剩 u1/u2 2 条，稳定
    # 结果 {A,B,D} × {u1,u2}——只有逐轮迭代才能正确收敛
    items, users = k_core_sets(df, 2)
    assert items == {"A", "B", "D"}
    assert users == {"u1", "u2"}


# ---------------------------------------------------------------- 统计


def test_compute_stats_synthetic():
    kcore_items = {"A", "B"}
    item_flags = {
        "A": {"has_title": True, "has_desc": False, "has_image": True, "n_images": 2},
        "B": {"has_title": True, "has_desc": True, "has_image": False, "n_images": 0},
    }
    counts = pd.Series({"A": 100, "B": 50})
    review_flags = {("u1", "A"): True, ("u1", "B"): False, ("u2", "A"): True}
    user_sequences = {"u1": ["A", "B"], "u2": ["A"]}

    stats = compute_stats(item_flags, review_flags, counts, kcore_items, user_sequences)
    ic = stats["item_coverage"]
    pd_ = stats["pattern_distribution"]
    pm = stats["popularity_missingness"]
    sc = stats["sequence_completeness"]

    assert ic["pct_title"] == 100.0
    assert ic["pct_image"] == 50.0
    assert ic["pct_description"] == 50.0
    assert ic["pct_none"] == 0.0
    assert ic["median_image_count"] == 1  # [0, 2] 中位
    assert ic["n_interactions"] == 150

    # 模式（含 desc）：A(u1/u2, 有文本+图, 无desc) -> text+image；B(u1, 无文本无图, 有desc) -> desc
    assert pd_["patterns"]["text+image"] == 2
    assert pd_["patterns"]["desc"] == 1
    assert pd_["patterns"]["none"] == 0
    assert pd_["has_text_share"] == round(2 / 3 * 100, 2)
    assert pd_["distinct"] == 2

    # 流行度×image：A(count100,图) vs B(count50,无图) -> 正相关
    assert pm["point_biserial_img"] > 0
    # 流行度×desc：A(count100,无desc) vs B(count50,有desc) -> 负相关
    assert pm["point_biserial_desc"] < 0

    # 序列完整性：u1 一半有图(0.5)，u2 全有图(1.0) -> mean=0.75
    assert sc["mean"] == 75.0
    assert sc["pct_below_half"] == 0.0


def test_compute_stats_empty_items():
    stats = compute_stats({}, {}, pd.Series(dtype=int), set(), {})
    assert stats["item_coverage"]["pct_image"] == 0.0
    assert stats["pattern_distribution"]["shares"]["text+image"] == 0.0


def test_point_biserial_known():
    # 完美单调分离 -> 强正相关（n=4 时 = 0.894）
    r = _point_biserial([0.0, 0.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    assert abs(r - 0.8944271909999159) < 1e-9
    # 常量 binary -> 0
    assert _point_biserial([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert _point_biserial([], []) == 0.0


def test_mnar_verdict_degenerate_image():
    # image 覆盖 100% -> 相关 0（无方差，无信息）；必须按 desc 判，且 0.118 属弱-中而非"强"
    ch, corr, label = _mnar_verdict({
        "point_biserial_img": 0.0,
        "point_biserial_desc": 0.1177,
        "point_biserial_text": 0.2208,
    })
    assert ch == "desc"
    assert abs(corr - 0.1177) < 1e-9
    assert label.startswith("弱-中")  # 0.118 属弱-中档（<0.15），不得判成"中"


def test_mnar_verdict_weak_and_fallback():
    # 全部接近 0 -> 弱
    ch, corr, label = _mnar_verdict({
        "point_biserial_img": 0.0, "point_biserial_desc": 0.0, "point_biserial_text": 0.0,
    })
    assert label == "弱（倾向接近随机）"
    # desc 为常数(0) -> 回退到 text
    ch, corr, label = _mnar_verdict({
        "point_biserial_img": 0.0, "point_biserial_desc": 0.0, "point_biserial_text": 0.32,
    })
    assert ch == "text" and abs(corr - 0.32) < 1e-9
    assert "中" in label


def test_iter_jsonl_gz_sample(tmp_path):
    path = tmp_path / "x.jsonl.gz"
    rows = [{"i": i, "parent_asin": f"P{i}"} for i in range(5)]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    got = list(iter_jsonl(path, sample=3))
    assert [r["i"] for r in got] == [0, 1, 2]
    assert len(list(iter_jsonl(path))) == 5
