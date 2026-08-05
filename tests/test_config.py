from cardterms.config import ExperimentConfig
from cardterms.db import healthcheck


def test_base_config_loads():
    cfg = ExperimentConfig.from_yaml("configs/base.yaml")
    assert cfg.chunking.chunk_tokens == 512
    assert cfg.retrieval.mode == "dense"


def test_invalid_config_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExperimentConfig(
            embedding={"model_name": "x"},
            chunking={"chunk_tokens": -5},  # must be rejected
        )


def test_database_reachable():
    assert healthcheck()
