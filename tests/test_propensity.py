import numpy as np

from mrseqrec.missing.propensity import fit_logistic, missing_probability


def _synth(beta_true, n=40000, seed=0):
    """按给定真值系数生成合成数据（可复现）。"""
    rng = np.random.default_rng(seed)
    logcount = rng.normal(0.0, 1.0, n)
    cat = rng.integers(0, 2, n).astype(float)
    X = np.column_stack([logcount, cat])
    eta = beta_true[0] + logcount * beta_true[1] + cat * beta_true[2]
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    y = (rng.random(n) < p).astype(float)
    return X, y


def test_fit_logistic_recovers_parameters():
    # §2.6 定式：logcount 系数负（越冷门越缺），β̂ 应还原为负
    beta_true = np.array([0.5, -1.2, 0.8])
    X, y = _synth(beta_true)
    beta = fit_logistic(X, y)
    assert np.max(np.abs(beta - beta_true)) < 0.15
    assert beta[1] < 0


def test_fit_logistic_degenerate_y_stays_finite():
    # 全 0 / 全 1 输出：无正则时 IRLS 发散，L2 应保住系数有界
    X = np.random.default_rng(0).normal(size=(100, 2))
    b0 = fit_logistic(X, np.zeros(100))
    b1 = fit_logistic(X, np.ones(100))
    assert np.all(np.isfinite(b0)) and np.all(np.isfinite(b1))
    assert np.max(np.abs(b0)) < 10 and np.max(np.abs(b1)) < 10


def test_missing_probability_bounds_and_monotone():
    X = np.column_stack([np.linspace(-3, 3, 100), np.ones(100)])
    beta = np.array([0.0, 1.0, 0.0])
    p = missing_probability(beta, X)
    assert np.all((p > 0) & (p < 1))
    assert np.all(np.diff(p) > 0)  # 随 logcount 增加单调增（此处 β>0 演示性质）
