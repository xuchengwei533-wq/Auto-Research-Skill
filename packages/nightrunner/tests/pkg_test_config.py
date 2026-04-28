from pathlib import Path

from nightrunner.config import DEFAULT_CONFIG, load_config, write_default_config_if_missing


def test_create_default_config(tmp_path: Path) -> None:
    cfg_path = write_default_config_if_missing(tmp_path)
    assert cfg_path.exists()
    loaded = load_config(tmp_path)
    assert loaded["project"]["name"] == tmp_path.name
    assert "train.py" in loaded["files"]["editable"]
    assert loaded["agent"]["base_url"] == "https://api.deepseek.com"
    assert loaded["agent"]["api_key_env"] == "DEEPSEEK_API_KEY"


def test_load_yaml_config(tmp_path: Path) -> None:
    custom = tmp_path / "nightrunner.yaml"
    custom.write_text(
        "project:\n  name: demo\nmetric:\n  name: val_loss\n  lower_is_better: true\n",
        encoding="utf-8",
    )
    loaded = load_config(tmp_path)
    assert loaded["project"]["name"] == "demo"
    assert loaded["metric"]["name"] == "val_loss"
