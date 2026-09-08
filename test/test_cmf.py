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


@pytest.mark.parametrize("name", ["lstm", "gru", "tcn", "transformer", "s4d", "mamba"])
def test_builtins_satisfy_contract(name):
    info = check_model(resolve_model(name), 8, T=32)
    assert info["causal"] and info["n_params"] > 0


def test_selective_scan_matches_loop():
    from cmf.models import selective_scan

    a, b = torch.rand(2, 37, 3, 4), torch.randn(2, 37, 3, 4)
    h, ref = torch.zeros(2, 3, 4), []
    for t in range(37):
        h = a[:, t] * h + b[:, t]
        ref.append(h)
    assert torch.allclose(selective_scan(a, b, chunk=8), torch.stack(ref, 1), atol=1e-5)


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


def test_shards_split_and_merge(tmp_path):
    out = tmp_path / "sharded"
    for i in range(3):
        cmf.run("lstm", suite=TINY, sizes=[4], out_dir=out, shard=(i, 3))
    assert len(list(out.glob("results.shard-*-of-3.csv"))) == 3
    df, _ = load_runs([out])
    assert len(df) == 4 and df.groupby(["memory_function", "alpha"]).size().eq(1).all()
    # an unsharded resume sees every shard's trials as finished
    cmf.run("lstm", suite=TINY, sizes=[4], out_dir=out)
    assert not (out / "results.csv").exists()


def test_report_with_bundled_baselines_and_pdf(tmp_path):
    from cmf.reference import available_references

    if "smoke" not in available_references():
        pytest.skip("no bundled smoke references")
    smoke = load_suite("smoke")
    run = cmf.run(Causal, name="Causal", suite=smoke, sizes=[8], out_dir=tmp_path / "c")
    out = cmf.report([run], out=tmp_path / "r.html", baselines="all", pdf=tmp_path / "r.pdf")
    html = out.read_text()
    assert "bundled baseline" in html and "LSTM" in html
    assert (tmp_path / "r.pdf").read_bytes()[:5] == b"%PDF-"


def test_ray_backend(tmp_path):
    pytest.importorskip("ray")
    run = cmf.run("lstm", suite=TINY, sizes=[4], out_dir=tmp_path / "ray", backend="ray",
                  ray_options={"cpus_per_trial": 1})
    assert len(load_runs([run])[0]) == 4
