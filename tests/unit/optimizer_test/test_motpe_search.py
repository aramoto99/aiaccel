import warnings
from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.optimizer.motpe_optimizer import MOTpeOptimizer
from tests.base_test import BaseTest


class TestMOTpeOptimizer(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_optimizer(self, data_dir, create_tmp_config):
        self.data_dir = data_dir
        self.optimizer = MOTpeOptimizer(self.load_config_for_test(self.configs["config_motpe.json"]))
        yield
        self.optimizer = None

    def test_is_startup_trials(self):
        assert self.optimizer.is_startup_trials()

    def test_generate_parameter(self):
        assert len(self.optimizer.generate_parameter()) > 0

        # if ((not self.is_startup_trials()) and (len(self.parameter_pool) >= 1))
        with patch.object(self.optimizer, "check_result", return_value=None):
            with patch.object(self.optimizer, "is_startup_trials", return_value=False):
                with patch.object(self.optimizer, "parameter_pool", [{}, {}, {}]):
                    assert self.optimizer.generate_parameter() is None

        # if len(self.parameter_pool) >= self.config.num_workers.get()
        self.optimizer.config.resource.num_workers = 0
        with patch.object(self.optimizer, "is_startup_trials", return_value=False):
            assert self.optimizer.generate_parameter() is None

    # def test_generate_initial_parameter(self, create_tmp_config):
    #     (self.optimizer.workspace.path / 'storage' / 'storage.db').unlink()
    #     config = self.optimizer.config.copy()
    #     self.config_motpe_path = create_tmp_config(self.data_dir / 'config_motpe_no_initial_params.json')
    #     optimizer = MOTpeOptimizer(self.optimizer.config)

    #     optimizer.__init__(config)
    #     assert len(optimizer.generate_initial_parameter()) > 0
    #     assert len(optimizer.generate_initial_parameter()) > 0

    def test_create_study(self):
        assert self.optimizer.create_study() is None

    def testserialize(self):
        self.optimizer.create_study()
        self.optimizer.trial_id.initial(num=0)
        self.optimizer.storage.state.set_any_trial_state(trial_id=0, state="ready")
        self.optimizer._rng = np.random.RandomState(0)
        assert self.optimizer.serialize(trial_id=0) is None

    def testdeserialize(self):
        self.optimizer.trial_id.initial(num=0)
        self.optimizer.storage.state.set_any_trial_state(trial_id=0, state="finished")
        self.optimizer.serialize(trial_id=0)
        assert self.optimizer.deserialize(trial_id=0) is None
