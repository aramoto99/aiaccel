import pytest

from aiaccel.manager import AbciManager, LocalManager, PylocalManager, create_manager


def test_create():
    assert create_manager("abci") == AbciManager
    assert create_manager("local") == LocalManager
    assert create_manager("python_local") == PylocalManager
    with pytest.raises(ValueError):
        assert create_manager("invalid")
