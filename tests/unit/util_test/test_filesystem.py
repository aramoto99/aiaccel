import shutil
from pathlib import Path

from aiaccel.storage import Storage
from aiaccel.util import create_yaml, file_create, file_delete, load_yaml, make_directories, make_directory
from aiaccel.workspace import Workspace


def test_create_yaml(clean_work_dir, work_dir):
    clean_work_dir()
    dict_lock = work_dir.joinpath('lock')
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    assert create_yaml(path, {}) is None
    file_delete(path)
    assert create_yaml(path, {}, dict_lock) is None


def test_file_create(clean_work_dir, work_dir):
    clean_work_dir()
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    file_create(path, 'hello')
    assert path.exists()


def test_load_yaml(clean_work_dir, work_dir):
    clean_work_dir()
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    dict_lock = work_dir.joinpath('lock')
    create_yaml(path, {})
    assert load_yaml(path) == {}
    assert load_yaml(path, dict_lock) == {}


def test_make_directory(clean_work_dir, work_dir):
    clean_work_dir()
    dict_lock = work_dir.joinpath('lock')
    assert make_directory(work_dir.joinpath('result')) is None
    assert make_directory(work_dir.joinpath('new'), dict_lock) is None
    assert make_directory(work_dir.joinpath('new'), dict_lock) is None  # exists


def test_make_directories(clean_work_dir, work_dir):
    clean_work_dir()
    dict_lock = work_dir.joinpath('lock')
    ds = [
        work_dir.joinpath('hp', 'ready'),
        work_dir.joinpath('hp', 'exist'),
        work_dir.joinpath('hp', 'new')
    ]

    if work_dir.joinpath('hp', 'exist').is_dir():
        work_dir.joinpath('hp', 'exist').rmdir()

    create_yaml(work_dir.joinpath('hp', 'exist'), {})
    assert make_directories(ds) is None
    shutil.rmtree(work_dir.joinpath('hp', 'exist'))
    file_create(work_dir.joinpath('hp', 'exist'), 'hello', dict_lock)
    assert make_directories(ds, dict_lock) is None
