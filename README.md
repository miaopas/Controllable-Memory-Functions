<div align="center">

# Controllable Memory Functions

**A synthetic benchmark that measures *what kinds of memory* a sequence model can learn.**

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>

</div>

> **Status: research code, becoming a tool.** Today you can reproduce the paper
> experiments and benchmark a new model by adding a few config files (see
> [Benchmark your own model](#benchmark-your-own-model)). We are working towards a one-command
> workflow, *bring a model → get a comparison report* (see [Roadmap](#roadmap)).

---

## Why

Sequence models (RNNs, CNNs, Transformers, state-space models) are often
compared on downstream tasks, which mix many effects together. This project
isolates one question: **how well can a model learn a target whose dependence on
the past follows a given *memory function* ρ(s)?** We control the *shape* of the
memory (exponential, polynomial, impulse, oscillating) and its *difficulty* (how far
back it reaches), then train models of different sizes and see where each one
breaks.

## The task

Inputs $x_t$ are random sequences. Targets are a (nonlinear) convolution of the
input with a memory function $\rho$:

$$
y_t \;=\; \sigma_{\text{out}}\!\Big(\sum_{s=0}^{t} \rho(s)\,\sigma_{\text{in}}(x_{t-s})\Big)
$$

The model sees $x_{1:t}$ and must predict $y_t$ at every step (causal
sequence-to-sequence regression, MSE loss). Each memory function has a
**difficulty parameter α**: larger α means longer or harder memory.

| Memory function | ρ(s) | What α controls | Name in code |
|---|---|---|---|
| Exponential | $e^{-s/\alpha}$ | decay time-scale | `exponential` |
| Polynomial | $(1+s)^{-1/\alpha}$ | tail heaviness | `polynomial` |
| Impulse | $\mathbb{1}[s = \lfloor \alpha T/2 \rfloor]$ | delay of a single recall | `impulse` |
| Airy | $\mathrm{Ai}\big(40(s/T - \alpha/2)\big)$ | how far the oscillation reaches | `airy` |
| Prony series | $\tfrac1K\sum_k \lambda_k^{s}$ | spread of time-scales (γ, K) | `prony` |
| Mixture | $(1-\beta)\,\rho_{\exp} + \beta\,\rho_{\text{imp}}$ | mixture weight β | `mix_exp_impulse` |

<p align="center">
  <img src="docs/assets/memory_functions.png" width="640" alt="The four main memory functions for increasing difficulty alpha">
  <br><em>The four main memory functions, coloured by normalised difficulty α (blue = easy, red = hard).</em>
</p>

All memory functions are defined in [`src/data/lf/rho.py`](src/data/lf/rho.py)
and the data generator is in [`src/data/lf/lf_datamodule.py`](src/data/lf/lf_datamodule.py).

## Models included

| Family | Models | Code |
|---|---|---|
| Recurrent | LSTM, GRU, linear / diagonal / gated RNN | `src/modules/layers/{rnn,gru,diagRNN}.py`, `torch.nn.LSTM` |
| Convolutional | TCN, dilated CNN | `src/modules/layers/tcn.py` |
| Attention | Transformer, relative-position Transformer | `src/modules/layers/{transformer,RPETransformer}.py` |
| State space | S4 / S4D, Mamba | `src/modules/layers/{s4,multi_s4,mamba}.py` |
| Mixed | any two of the above stacked (e.g. `LSTM_S4`) | `configs/module/lit_module/mix_model/` |

Every model is wrapped as `Linear(1→m) → model → Linear(m→1)`, where *m* is the
model width that the sweeps vary.

## What you get

For each model, a sweep over **memory function × α × model size m × seed** gives
the minimum training/validation loss per setting. From that we plot:

- **loss vs α** for each memory function: how quickly performance degrades as memory gets longer;
- **loss vs model size**: whether a bigger model fixes it;
- a **difficulty-sensitivity score**: the relative loss increase from the easiest to the hardest α.

The planned report puts these side by side with reference models.

---

## Installation

Python 3.10 or 3.11. A CUDA GPU is strongly recommended for the full sweeps.

```bash
git clone https://github.com/miaopas/Controllable-Memory-Functions
cd Controllable-Memory-Functions
```

**Option A: Poetry (pinned versions, recommended)**

```bash
poetry install --no-root
```

Then run the commands below with `poetry run python …`.

**Option B: pip**

```bash
pip install -r requirements.txt
```

**Option C: container (HPC)**: build `apptainer1.def`. See
[`scripts/helpers.sh`](scripts/helpers.sh).

Some models need extra compiled packages: S4 can use `pykeops` or the CUDA
kernels in `extensions/kernels/`, and Mamba needs `mamba-ssm`. LSTM, TCN and
Transformer work with plain PyTorch.

### Logging

Sweeps log to [Weights & Biases](https://wandb.ai) by default. Use your own
account, and set the `entity` in [`configs/logger/wandb.yaml`](configs/logger/wandb.yaml):

```bash
export WANDB_API_KEY=<your-key>   # or: wandb login
```

For a single run you can avoid W&B with `logger=csv`.

## Quick start

All entry points use [Hydra](https://hydra.cc), so any config value can be
overridden from the command line.

**1. One training run**: an LSTM of width 16 on the exponential memory, α = 0.2:

```bash
python src/train.py data=lf module=lit_module/LSTM \
  data.rho_name=exponential data.rho_param.alpha=0.2 \
  trainer=cpu trainer.max_epochs=5 logger=csv +skip_confirm=true
```

**2. A full sweep** over α × width × seed (uses [Ray Tune](https://docs.ray.io/en/latest/tune/)):

```bash
python src/train_multiple.py experiment=Dataset/LSTM/exp/coarse trainer=gpu +skip_confirm=true
```

The sweep grid is defined in the experiment file under `sweeper.param`.
When it finishes, the results are pulled from W&B into
`results/<project>_data_clean.pkl`, and a summary PDF report is written next to it.

**3. On a PBS cluster**: each line of a script in `scripts/` is one sweep:

```bash
export WANDB_API_KEY=<your-key>
./_submit_jobs.sh scripts/Dataset/LSTM.sh scripts/Dataset/S4.sh
```

Set your PBS project ID in `_run_array_jobs.pbs` first.

### Experiment catalogue

| Folder | Study |
|---|---|
| `configs/experiment/Dataset/<Model>/<function>/coarse.yaml` | main study: single-layer model, α × width × seed |
| `…/<function>/multi_layer*.yaml` | the same with deeper models |
| `configs/experiment/Dataset/<Model>/{prony_K,prony_gamma,mix_target}/` | Prony-series and mixture targets |
| `configs/experiment/Mixed_Models/<A>_<B>/` | two-layer hybrids (e.g. LSTM → S4) |
| `scripts/Dataset_length_ablation/` | sequence-length ablation (T = 64 … 2048) |

## Benchmark your own model

Until the one-command tool lands (see [Roadmap](#roadmap)), adding a model
takes four small steps. Use LSTM as the template.

**1. Write the layer.** Input and output have shape `(batch, T, m)`, and it must be
**causal**: the output at *t* may only use inputs up to *t*. Accept `**kwargs` in
`forward`. Returning a tuple is fine; only the first element is used.

```python
# src/modules/layers/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, d_model: int = 16):
        super().__init__()
        self.rnn = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x, **kwargs):
        return self.rnn(x)[0]
```

**2. Add a layer config** `configs/module/layers/MyModel.yaml`:

```yaml
_target_: src.modules.layers.my_model.MyModel
d_model: 16
```

**3. Add a module config** `configs/module/lit_module/MyModel.yaml`. Copy
`LSTM.yaml` and replace the `LSTM` layer with `MyModel`, wiring `Input.output_size`,
`MyModel.d_model` and `Output.input_size` to the same width.

**4. Add experiment configs.** Copy `configs/experiment/Dataset/LSTM/` to
`configs/experiment/Dataset/MyModel/`. Set `override /module: lit_module/MyModel`
in `base.yaml`, and in each `<function>/coarse.yaml` change the swept width key to
`module.layers.MyModel.d_model` and update the W&B `project` name. Then run:

```bash
python src/train_multiple.py experiment=Dataset/MyModel/exp/coarse trainer=gpu +skip_confirm=true
```

To get the automatic PDF report and reuse the `paper_plots/` scripts, register the
width key (`'MyModel': 'module.layers.MyModel.d_model'`) in both
[`src/utils/model_registry.py`](src/utils/model_registry.py) and
[`paper_plots/model_registry.py`](paper_plots/model_registry.py).

## Reproducing the paper figures

`paper_plots/<n>_<study>/` holds one folder per figure group, each with a
`generate_*.py` script and the resulting PDFs. The scripts read processed sweep
data (`results/*_data_clean.pkl`), which is **not included** in this repository
because of its size. To regenerate it, run the corresponding sweeps and export
them from W&B (see `notebooks/Dataset/step1/get_data.py`).

## Repository layout

```
├── configs/                Hydra configs
│   ├── data/                 lf.yaml: the memory-function dataset
│   ├── module/               layers/, lit_module/ (model + optimiser), model/
│   ├── experiment/           sweep definitions (Dataset/, Mixed_Models/, Train/)
│   ├── trainer/ logger/ callbacks/ paths/ hydra/ …
│   ├── train.yaml            single run
│   └── train_multiple.yaml   Ray Tune sweep
├── src/
│   ├── data/lf/              rho.py (memory functions), lf_datamodule.py (data)
│   ├── modules/              layers/ (all models), models/sequential.py, lit_module.py
│   ├── utils/                Hydra helpers, result_processing.py (W&B → results + PDF)
│   ├── train.py              single-run entry point
│   └── train_multiple.py     sweep entry point
├── scripts/                job lists for cluster submission
├── paper_plots/            figure scripts and PDFs from the paper
├── notebooks/              exploratory analysis
├── extensions/kernels/     optional CUDA kernels for S4 (Cauchy / Vandermonde)
├── _submit_jobs.sh, _run_array_jobs.pbs   PBS array-job helpers
└── apptainer1.def          container definition
```

## Roadmap

1. **Phase 0, demo:** a `quick` suite that runs on one GPU in about an hour, a local
   runner without Ray or W&B, and an HTML comparison report.
2. **Phase 1, tool:** a `cmf run` / `cmf report` CLI, a causality check for user
   models, and shipped reference results for LSTM, TCN, Transformer, S4 and Mamba.
3. **Phase 2, research:** user-defined memory functions, per-model learning-rate
   search, and multi-layer and hybrid suites.

Contributions and issues are welcome.

## Citation

If you use this benchmark, please cite the accompanying paper (reference coming
soon).

## License

Released under the [MIT License](LICENSE). `src/modules/layers/s4.py` and
`extensions/kernels/` are adapted from
[state-spaces/s4](https://github.com/state-spaces/s4) (Apache-2.0);
`extensions/kernels/map.h` keeps its original notice. The project layout is based on
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
