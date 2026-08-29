from mrseqrec.data.synthetic import generate_interactions
from mrseqrec.pipeline import run_training
from mrseqrec.utils.config import Config


def test_pipeline_end_to_end(tmp_path):
    df = generate_interactions(n_users=120, n_items=60, min_len=5, max_len=15, seed=7)
    f = tmp_path / "data.txt"
    df.to_csv(f, sep=" ", header=False, index=False)

    config = Config.model_validate(
        {
            "data": {"interactions_file": str(f), "min_interactions": 5, "max_seq_len": 10, "num_negatives": 20},
            "model": {"hidden_dim": 16, "num_layers": 1, "num_heads": 2},
            "train": {"batch_size": 32, "epochs": 2, "device": "cpu"},
            "eval": {"topk": [5, 10], "batch_size": 32},
        }
    )
    result = run_training(config)
    assert "test/recall@5" in result.metrics
    assert 0.0 <= result.metrics["test/recall@5"] <= 1.0
    assert result.retention is not None
    assert result.retention.auc >= 0.0
