import pytest

from mrseqrec.utils.config import Config


def _base() -> dict:
    return {"data": {"interactions_file": "x.txt"}, "model": {}, "train": {}, "eval": {}}


def test_defaults():
    c = Config.model_validate(_base())
    assert c.model.hidden_dim == 64
    assert c.train.epochs == 50
    assert c.eval.topk == [10, 20]


def test_rejects_invalid_min_interactions():
    with pytest.raises(Exception):
        Config.model_validate({**_base(), "data": {"interactions_file": "x.txt", "min_interactions": 0}})


def test_rejects_unknown_model():
    with pytest.raises(Exception):
        Config.model_validate({**_base(), "model": {"name": "nope"}})
