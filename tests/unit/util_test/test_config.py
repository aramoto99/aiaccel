import pytest

from aiaccel.config import is_multi_objective, load_config


def test_load_config(config_yaml):
    yaml_config = load_config(config_yaml)
    # assert json_config.generic.workspace == "/tmp/work"
    # assert yaml_config.generic.workspace == "./hoge"
    assert yaml_config.generic.workspace == "/tmp/work"

def test_config_not_exists():
    with pytest.raises(ValueError):
        load_config("invalid")


def test_is_multi_objective(config_yaml):
    config = load_config(config_yaml)
    assert is_multi_objective(config) is False
