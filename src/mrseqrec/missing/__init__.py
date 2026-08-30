"""缺失建模子包（S2 最小核：MNAR 倾向模型 + 缺失环境采样，关口 2 §2.6）。"""

from mrseqrec.missing.propensity import fit_logistic, missing_probability
from mrseqrec.missing.sampler import (
    env_distribution,
    env_id,
    mcar_corrupt,
    mnar_select,
    sample_missingness,
)

__all__ = [
    "fit_logistic",
    "missing_probability",
    "env_id",
    "env_distribution",
    "mcar_corrupt",
    "mnar_select",
    "sample_missingness",
]
