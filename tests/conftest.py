import pytest

from trackml_edge.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    """Small synthetic TrackML-format dataset (8 events) shared by all tests."""
    return make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_events=8, n_particles=120, n_noise=150)
