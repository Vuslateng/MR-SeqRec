import json
import sys

import numpy as np

from scripts.estimate_propensity import load_meta_items, propensity_report


def _meta_file(tmp_path, rows: list[dict], name="meta.jsonl"):
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


def _nm_items(n=30000, seed=0):
    """真值 β=(0.3, -1.4)：logcount 越高越不可能缺 desc（β̂ 应显著为负）。"""
    rng = np.random.default_rng(seed)
    logc = rng.normal(0.0, 1.0, n)
    eta = 0.3 + logc * -1.4
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    y = (rng.random(n) < p).astype(float)
    return logc, y


def test_load_meta_items_parses():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = _meta_file(Path(d), [
            {"parent_asin": "A1", "count": 100, "has_desc": False},
            {"parent_asin": "A2", "count": 0, "has_desc": True},
            {"parent_asin": "A3", "count": 5, "has_desc": False},
        ])
        logc, missing = load_meta_items(p)
        assert missing.tolist() == [1.0, 0.0, 1.0]
        assert np.allclose(logc, [np.log1p(100), 0.0, np.log1p(5)])


def test_load_meta_items_requires_fields(tmp_path):
    p = _meta_file(tmp_path, [{"parent_asin": "A1", "count": 3}])  # 缺 has_desc
    import pytest

    with pytest.raises(AssertionError):
        load_meta_items(p)


def test_load_meta_items_empty_raises(tmp_path):
    import pytest

    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="为空"):
        load_meta_items(p)


def test_propensity_report_mnar_direction():
    """强 MNAR（β̂<0）：CI 不含 0、wald p 显著、判读命中断言成立。"""
    logc, y = _nm_items()
    r = propensity_report(logc, y)
    lg = r["logistic"]
    assert lg["beta_log_count"] < 0
    assert lg["ci95_log_count"][1] < 0  # CI 整体为负
    assert lg["wald_p"] is not None and lg["wald_p"] < 1e-6
    assert r["verdict"].startswith("MNAR 倾向成立")


def test_propensity_report_random_gives_weak():
    """缺失与流行度无关：斜率≈0，CI 含 0，判读为倾向弱（不误报 MNAR）。"""
    rng = np.random.default_rng(1)
    logc = rng.normal(0.0, 1.0, 50000)
    y = (rng.random(50000) < 0.5).astype(float)
    r = propensity_report(logc, y)
    lo, hi = r["logistic"]["ci95_log_count"]
    assert lo < 0 < hi  # CI 含 0
    assert r["verdict"].startswith("倾向弱")


def test_propensity_report_mnar_degenerate_counts():
    """全 0（全 desc 完整）退化：L2 保住有界，CI 含 0 判弱，不抛错。"""
    logc = np.linspace(0.0, 6.0, 500)
    y = np.zeros(500)
    r = propensity_report(logc, y)
    assert all(np.isfinite(v) for v in [r["logistic"]["beta_log_count"],
                                        r["logistic"]["beta_log_count_se"]])
    assert r["logistic"]["ci95_log_count"][0] < 0 < r["logistic"]["ci95_log_count"][1]


def test_cli_smoke(tmp_path):
    import scripts.estimate_propensity as mod

    rng = np.random.default_rng(3)
    logc = rng.normal(0.0, 1.0, 2000)
    y = (rng.random(2000) < 0.5).astype(float)
    rows = [{"parent_asin": f"A{i}", "count": int(round(np.expm1(logc[i]))),
             "has_desc": bool(y[i] == 0.0)} for i in range(2000)]
    meta = _meta_file(tmp_path, rows)
    out = tmp_path / "propensity.json"
    old = sys.argv
    sys.argv = ["estimate_propensity", "--meta-items", str(meta), "--out", str(out)]
    try:
        mod.main()
    finally:
        sys.argv = old
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "logistic" in data and "verdict" in data
    assert data["n_items"] == 2000
