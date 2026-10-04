import pytest

from speechlab import audio, models, store


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    for folder in ("clips", "features", "axes", "results"):
        (tmp_path / folder).mkdir()
    for module in (audio, models, store):
        monkeypatch.setattr(module, "DATA", tmp_path)
    return tmp_path


@pytest.fixture
def clips():
    from demos.fixtures import seed

    return seed()
