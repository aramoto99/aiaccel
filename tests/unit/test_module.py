import asyncio
import logging
import time

import numpy as np
import pytest

from aiaccel.module import AbstractModule
from aiaccel.util import str_to_logging_level
from tests.base_test import BaseTest


async def async_function(func):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, func)


async def delay_make_directory(sleep_time, d):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, time.sleep, sleep_time)
    d.mkdir()


def dummy_break():
    import sys

    sys.exit()


class TestAbstractModule(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_module(self, clean_work_dir):
        self.module = AbstractModule(self.load_config_for_test(self.configs["config.json"]), "abstract")
        self.module.logger = logging.getLogger(__name__)
        yield
        self.module = None

    def test_set_logger(self, work_dir):
        assert (
            self.module.set_logger(
                "root.optimizer",
                work_dir.joinpath(self.module.workspace.log, self.module.config.logger.file.optimizer),
                self.module.config.logger.log_level.optimizer,
                self.module.config.logger.stream_level.optimizer,
                "Optimizer",
            )
            is None
        )

    def test_pre_process(self):
        try:
            self.module.pre_process()
            assert False
        except NotImplementedError:
            assert True

    def test_post_process(self):
        try:
            self.module.post_process()
            assert False
        except NotImplementedError:
            assert True

    def testserialize(self):
        self.module._rng = np.random.RandomState(0)
        assert self.module.serialize(0) is None

    def testdeserialize(self):
        self.module._rng = np.random.RandomState(0)
        self.module.serialize(1)
        assert self.module.deserialize(1) is None

    def test_is_error_free(self):
        assert self.module.is_error_free() is True

    def test_resume(self):
        try:
            assert self.module.resume() is None
            assert False
        except NotImplementedError:
            assert True
