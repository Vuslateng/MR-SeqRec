import pytest

from mrseqrec.eval.retention import compute_retention


def test_compute_retention():
    report = compute_retention({0.0: 0.5, 0.5: 0.4, 1.0: 0.3})
    assert report.retention_by_rho == {0.0: 1.0, 0.5: 0.8, 1.0: 0.6}
    assert report.auc > 0.0


def test_zero_base_raises():
    with pytest.raises(ValueError):
        compute_retention({0.0: 0.0, 0.5: 0.1})
