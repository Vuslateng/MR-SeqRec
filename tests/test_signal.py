from mrseqrec.data.signal import interval_stats, repurchase_check
from mrseqrec.data.synthetic import generate_interactions


def test_interval_stats_positive():
    df = generate_interactions(n_users=100, n_items=50, min_len=5, max_len=20, seed=1)
    stats = interval_stats(df)
    assert stats.median_days > 0
    assert 0.0 <= stats.frac_short_cycle <= 1.0


def test_repurchase_detects_30day():
    df = generate_interactions(
        n_users=200, n_items=100, min_len=10, max_len=30,
        periodic_fraction=0.7, period_days=30.0, seed=2,
    )
    rep = repurchase_check(df)
    assert rep.healthy_signal
    assert rep.periodic_pairs >= 10
    assert 25.0 <= rep.dominant_period_days <= 40.0


def test_no_periodicity_without_repurchase():
    """无周期复购（随机游走）时不应误报。"""
    df = generate_interactions(n_users=200, n_items=100, min_len=10, max_len=30, periodic_fraction=0.0, seed=5)
    rep = repurchase_check(df)
    assert rep.periodic_pairs < 10  # 随机数据几乎不会产生 ≥3 次同商品复购
