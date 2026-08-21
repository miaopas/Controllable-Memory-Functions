"""Tests for the cmf benchmark tool. Run from the repository root: ``pytest test/test_cmf.py``.

They only need torch, numpy, scipy, pandas, matplotlib and pyyaml (no Lightning/Hydra/W&B).
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cmf  # noqa: E402
from cmf.analysis import best_over_seeds, sensitivity  # noqa: E402
from cmf.results import load_runs  # noqa: E402
from cmf.spec import ModelCheckError, check_model, from_callable, resolve_model  # noqa: E402
from cmf.suite import available_suites, load_suite  # noqa: E402

TINY = {
    "name": "tiny",
    "T": 16,
    "data": {"n_train": 64, "n_val": 16, "n_test": 16},
    "training": {"epochs": 2, "min_epochs": 1, "batch_size": 16},
    "seeds": [1],
    "functions": {"exponential": {"alpha": [0.1, 0.4]}, "impulse": {"alpha": [0.1, 1.0]}},
}


class Causal(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rnn = nn.GRU(width, width, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


class Bidirectional(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.rnn = nn.GRU(width, width // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.rnn(x)[0]


class WrongShape(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin = nn.Linear(width, width + 1)

    def forward(self, x):
        return self.lin(x)


def test_builtin_suites_load():
    assert {"smoke", "quick", "standard"} <= set(available_suites())
    for name in available_suites():
        s = load_suite(name)
        assert s.functions and all(f.alpha for f in s.functions)


@pytest.mark.parametrize("name", ["lstm", "gru", "tcn", "transformer"])
def test_builtins_satisfy_contract(name):
    info = check_model(resolve_model(name), 8, T=32)
    assert info["causal"] and info["n_params"] > 0


def test_causality_check_rejects_bidirectional():
    with pytest.raises(ModelCheckError, match="not causal"):
        check_model(from_callable("Bi", Bidirectional), 8)


def test_shape_check_rejects_wrong_width():
    with pytest.raises(ModelCheckError):
        check_model(from_callable("Bad", WrongShape), 8)


def test_import_path_with_size_arg():
    spec = resolve_model("examples.my_model:ResidualGRU", size_arg="width")
    assert spec.name == "ResidualGRU"
    assert check_model(spec, 8)["causal"]


def test_run_resume_and_report(tmp_path):
    run_a = cmf.run(Causal, name="Causal", suite=TINY, sizes=[4], out_dir=tmp_path / "a")
    run_b = cmf.run("lstm", suite=TINY, sizes=[4], out_dir=tmp_path / "b")
    df, metas = load_runs([run_a, run_b])
    assert len(df) == 2 * 4 and (df["status"] == "ok").all()
    assert {m["model"] for m in metas} == {"Causal", "LSTM"}

    # resuming does not re-run finished trials
    cmf.run(Causal, name="Causal", suite=TINY, sizes=[4], out_dir=tmp_path / "a")
    assert len(load_runs([run_a])[0]) == 4

    sens = sensitivity(best_over_seeds(df, "train_rloss_min"))
    assert set(sens["memory_function"]) == {"exponential", "impulse"}

    out = cmf.report([run_a, run_b], out=tmp_path / "report.html")
    html = out.read_text()
    assert "Memory benchmark: Causal" in html and html.count("data:image/png;base64") >= 4


def test_suite_mismatch_is_refused(tmp_path):
    cmf.run("lstm", suite=TINY, sizes=[4], out_dir=tmp_path / "r")
    other = dict(TINY, T=8)
    with pytest.raises(ValueError, match="different suite"):
        cmf.run("lstm", suite=other, sizes=[4], out_dir=tmp_path / "r")


def test_data_matches_definition():
    """Target equals the direct convolution y_t = tanh(sum_s rho(s) u_{t-s}).

    The stored model input is already u = tanh(x) for Gaussian noise x (inner activation).
    """
    from cmf.data import make_task
    from cmf.suite import DataConfig

    d = make_task("exponential", 0.4, 16, 3, DataConfig(n_train=4, n_val=1, n_test=1))
    u, y, rho = d.train.x[:, :, 0], d.train.y[:, :, 0], d.rho
    assert u.abs().max() < 1  # inputs went through tanh
    ref = torch.stack([torch.tanh(sum(rho[s] * u[:, t - s] for s in range(t + 1)))
                       for t in range(16)], dim=1)
    assert torch.allclose(y, ref, atol=1e-5)
