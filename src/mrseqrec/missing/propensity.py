"""缺失倾向模型（关口 2 §2.6，C1 部分）。

缺失指示 O 本身可观测（这条数据里有没有 desc、有没有 image 是知道的），
所以"缺失倾向只依赖可观测协变量"的那一半（识别条件 C1）可以监督估计——
即"缺失可识别"部分。纯 numpy IRLS 逻辑回归，不依赖 sklearn。

定式（§2.6）：P(O_i^desc=0 | x) = σ(α + β·log(count_i) + γ^T z_i)，β̂<0
（数据核查点二列 +0.1177 是"desc 存在"与流行度的相关；缺失事件与之反向，
系数符号取负：越冷门越缺 desc）。

本模块只保证「逻辑正确 + 参数还原」两层：真值恢复测试本地跑通，真实数据上的
倾向估计在服务器上同一批执行（与 image 探针同批）。
"""

from __future__ import annotations

import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """列首拼一列 1（截距项）。X: (n, d) → (n, d+1)。"""
    return np.column_stack([np.ones(X.shape[0]), X])


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> np.ndarray:
    """IRLS（Newton-Raphson）拟合逻辑回归 P(y=1 | X)。

    参数
    ----
    X : (n, d) 连续特征与 one-hot（调用方自行拼装，含可观测协变量）；
    y : (n,) ∈ {0,1}，对缺失指示取 1=缺失；
    l2 : L2 正则系数，防 one-hot 共线性并保证退化数据收敛（含截距一并正则，
         正则项随样本量被稀释，大样本下对系数几乎无偏）；
    max_iter / tol : 迭代上限与系数步长停机阈值。

    返回 (d+1,) 系数数组，下标 0 为截距。
    """
    Xb = add_bias(X)
    n, d = Xb.shape
    beta = np.zeros(d)
    for _ in range(max_iter):
        eta = Xb @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        w = mu * (1.0 - mu)
        H = Xb.T @ (Xb * w[:, None]) + l2 * np.eye(d)
        grad = Xb.T @ (y - mu) - l2 * beta
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            break
    return beta


def missing_probability(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    """由拟合系数预测缺失概率 P(缺失 | X)。返回 (n,)。"""
    eta = add_bias(X) @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
