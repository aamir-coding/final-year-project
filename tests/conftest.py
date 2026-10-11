import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.make_synthetic import make, synthetic_cfg  # noqa: E402
from mamg.config import load_config  # noqa: E402

REAL_META = ROOT / "data" / "raw" / "metadata.csv"


@pytest.fixture(scope="session")
def base_cfg():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg["paths"]["metadata_csv"] = str(REAL_META)
    return cfg


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_data")
    make(d, n_patients=120, seed=0)
    return d


@pytest.fixture()
def synth_cfg(base_cfg, synth_dir, tmp_path):
    return synthetic_cfg(base_cfg, synth_dir, tmp_path / "synthetic_runs")


real_data = pytest.mark.skipif(not REAL_META.exists(), reason="real PAD-UFES-20 metadata not downloaded")
