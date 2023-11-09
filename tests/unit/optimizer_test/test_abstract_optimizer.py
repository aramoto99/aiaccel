import asyncio
import os
import time
from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.optimizer import AbstractOptimizer
from tests.base_test import BaseTest


async def async_function(func):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, func)


async def make_directory(sleep_time, d):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    os.mkdir(d)


class TestAbstractOptimizer(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir):
        self.optimizer = AbstractOptimizer(self.load_config_for_test(self.configs["config.yaml"]))
        yield
        self.optimizer = None

    def test_register_new_parameters(self):
        params = [
            {"name": "x1", "type": "uniform_float", "value": 0.1},
            {"name": "x2", "type": "uniform_float", "value": 0.1},
        ]

        assert self.optimizer.register_new_parameters(params) is None

    def test_generate_initial_parameter(self):
        with patch.object(self.optimizer.params, "sample", return_value=[]):
            assert self.optimizer.generate_initial_parameter() == []

        p = [
            {"name": "x1", "type": "uniform_float", "value": 1.0},
            {"name": "x2", "type": "uniform_float", "value": 2.0},
        ]

        with patch.object(self.optimizer.params, "sample", return_value=p):
            assert self.optimizer.generate_initial_parameter() == [
                {"name": "x1", "type": "uniform_float", "value": 1.0},
                {"name": "x2", "type": "uniform_float", "value": 2.0},
            ]

    def test_generate_parameter(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = self.optimizer.generate_parameter()

    def test_generate_new_parameter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.optimizer, "num_of_generated_parameter", 0)
            m.setattr(self.optimizer, "generate_initial_parameter", lambda: None)
            assert self.optimizer.generate_new_parameter() is None

            m.setattr(self.optimizer, "num_of_generated_parameter", 1)
            m.setattr(self.optimizer, "generate_parameter", lambda: None)
            assert self.optimizer.generate_new_parameter() is None

    def test_is_error_free(self):
        self.optimizer.storage.error.all_delete()
        assert self.optimizer.is_error_free() is True

        self.optimizer.storage.error.set_any_trial_error(trial_id=0, error_message="test warning")
        assert self.optimizer.check_error() is True

        self.optimizer.storage.error.set_any_trial_exitcode(trial_id=0, exitcode=1)
        assert self.optimizer.check_error() is False

        self.optimizer.storage.error.all_delete()
        self.optimizer.config.generic.is_ignore_warning = False
        self.optimizer.storage.error.set_any_trial_error(trial_id=0, error_message="test warning")
        assert self.optimizer.check_error() is False

    def test_serialize(self):
        self.optimizer._rng = np.random.RandomState(0)
        assert self.optimizer.serialize(0) is None

    def test_deserialize(self):
        self.optimizer._rng = np.random.RandomState(0)
        self.optimizer.serialize(1)
        assert self.optimizer.deserialize(1) is None
