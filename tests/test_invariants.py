"""Repository-level guards for invariants 5, 8 and 9."""
import ast
import inspect
from pathlib import Path

import pandas as pd

import mamg.train
import mamg.xai_eval
from mamg.config import config_hash, output_dirs

ROOT = Path(__file__).resolve().parents[1]


def _calls_with_allow_test(module) -> list[int]:
    tree = ast.parse(inspect.getsource(module))
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
            for kw in n.keywords if kw.arg == "allow_test"
            and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)]


def test_train_and_xai_never_request_the_test_split():
    """Invariant 5: only mamg.evaluate (with --final-test) may pass allow_test."""
    for module in (mamg.train, mamg.xai_eval):
        assert _calls_with_allow_test(module) == [], module.__name__
        assert '"test"' not in inspect.getsource(module).replace('"test" if', ""), module.__name__


def test_synthetic_outputs_never_go_to_results(synth_cfg):
    """Invariant 9."""
    dirs = output_dirs(synth_cfg)
    assert dirs["label"] == "SYNTHETIC"
    for k in ("results", "checkpoints", "splits"):
        assert "synthetic_runs" in str(dirs[k])


def test_committed_results_contain_no_synthetic_rows():
    """Invariant 9: anything under results/ must come from real data."""
    for csv in (ROOT / "results").rglob("*.csv"):
        df = pd.read_csv(csv)
        if "data_kind" in df.columns:
            assert set(df["data_kind"].dropna()) <= {"REAL"}, csv
    assert not list((ROOT / "results").rglob("*SYNTHETIC*"))


def test_config_hash_ignores_paths_runtime_and_seed(base_cfg):
    import copy
    a = copy.deepcopy(base_cfg)
    b = copy.deepcopy(base_cfg)
    b["paths"]["metadata_csv"] = "/kaggle/input/x.csv"
    b["runtime"]["device"] = "cuda"
    b["seed"] = 999
    assert config_hash(a) == config_hash(b)
    b["train"]["lr"] = 1.0
    assert config_hash(a) != config_hash(b)
