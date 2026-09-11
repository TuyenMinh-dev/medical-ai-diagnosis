from src.utils.config import load_config


def test_load_config():
    cfg = load_config("configs/config.yaml")
    assert cfg["project_name"] == "medical-ai-diagnosis"
    assert "model" in cfg
