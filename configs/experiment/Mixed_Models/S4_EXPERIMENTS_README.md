# S4-Based Mixed Architecture Experiments

## Overview

This directory contains configuration files for S4-based mixed architecture experiments. These experiments benchmark how Structured State Space (S4) models perform when composed with other sequential architectures (S4, TCN) on synthetic long-memory tasks.

## Created Configurations

### 1. Module Configurations
Located in: `configs/module/lit_module/mix_model/`

- **S4_S4.yaml**: Two S4 layers in sequence (homogeneous baseline)
- **S4_TCN.yaml**: S4 layer followed by TCN (structured state → convolutional)
- **TCN_S4.yaml**: TCN layer followed by S4 (convolutional → structured state)

### 2. Architecture Details

#### S4_S4 (Homogeneous Baseline)
```
Linear(1→D) → S4(D, 1 layer) → S4(D, 1 layer) → Linear(D→1)
```
- **Purpose**: Baseline for two-layer S4 performance
- **Key parameter**: `model_size` controls `d_model` for both S4 layers
- **Hypothesis**: Deep S4 captures hierarchical temporal structure

#### S4_TCN (S4 → TCN)
```
Linear(1→D) → S4(D, 1 layer) → TCN(D, 10 layers) → Linear(D→1)
```
- **Purpose**: S4 extracts structured features, TCN refines with local convolutions
- **Key parameter**: `model_size` controls both `d_model` and TCN `channel`
- **Hypothesis**: S4's structured kernels + TCN's locality = best of both worlds

#### TCN_S4 (TCN → S4)
```
Linear(1→64) → TCN(64→D, 10 layers) → S4(D, 1 layer) → Linear(D→1)
```
- **Purpose**: TCN extracts local features, S4 integrates with structured memory
- **Key parameter**: `model_size` controls S4 `d_model` and TCN output `channel`
- **Note**: Input projection fixed at 64 (follows TCN_LSTM pattern)
- **Hypothesis**: Local feature extraction → structured integration

## Experiment Configuration

### Memory Patterns Tested
Each architecture is evaluated on 4 memory decay patterns:

1. **Exponential** (`exp`):
   - Formula: `ρ(t) = exp(-t/α)`
   - Sequence length: T=1024
   - Alpha range: 10 values from 0.053 to 0.429
   - Total runs: 10 × 5 × 4 = **200 runs**

2. **Polynomial** (`polynomial`):
   - Formula: `ρ(t) = 1/(1+t)^(1/α)`
   - Sequence length: T=1024
   - Alpha range: 10 values from 0.05 to 0.3
   - Total runs: 10 × 5 × 4 = **200 runs**

3. **Impulse** (`impulse`):
   - Formula: Box function centered at `α*T/2`
   - Sequence length: T=256 (shorter for harder task)
   - Alpha range: 8 values from 0.01 to 1.0
   - Uses ReduceLROnPlateau scheduler
   - Total runs: 8 × 5 × 4 = **160 runs**

4. **Airy** (`airy`):
   - Formula: Airy function `Ai(t/α)`
   - Sequence length: T=1024
   - Alpha range: 8 values from 0.01 to 1.0
   - Total runs: 8 × 5 × 4 = **160 runs**

### Hyperparameter Sweep

For each memory pattern, we sweep over:
- **Model size**: [8, 16, 32, 64, 128] (5 values)
- **Memory decay rate (α)**: 8-10 values (pattern-dependent)
- **Random seed**: [11, 22, 33, 44] (4 values)

**Total per architecture**: 200 + 200 + 160 + 160 = **720 runs**
**Total for all 3 architectures**: 720 × 3 = **2,160 runs**

## Execution Scripts

Located in: `scripts/Mixed_Models/`

### Individual Architecture Scripts
- **S4_S4.sh**: Run all S4_S4 experiments (720 runs)
- **S4_TCN.sh**: Run all S4_TCN experiments (720 runs)
- **TCN_S4.sh**: Run all TCN_S4 experiments (720 runs)

### Master Script
- **run_all_s4_experiments.sh**: Run all S4-based experiments sequentially (2,160 runs)

## Usage

### Run a Single Experiment
```bash
# Example: S4_TCN on exponential decay
python src/train_multiple.py trainer=gpu experiment=Mixed_Models/S4_TCN/exp/coarse +skip_confirm=true
```

### Run All Experiments for One Architecture
```bash
# Run all S4_TCN experiments
bash scripts/Mixed_Models/S4_TCN.sh
```

### Run All S4-Based Experiments
```bash
# Run all 2,160 experiments
bash scripts/Mixed_Models/run_all_s4_experiments.sh
```

## W&B Projects

Results are logged to separate W&B projects:

| Architecture | Memory Pattern | W&B Project Name |
|-------------|----------------|------------------|
| S4_S4 | Exponential | Mixed_S4_S4_exp_experiment |
| S4_S4 | Polynomial | Mixed_S4_S4_poly_experiment |
| S4_S4 | Impulse | Mixed_S4_S4_impulse_experiment |
| S4_S4 | Airy | Mixed_S4_S4_airy_experiment |
| S4_TCN | Exponential | Mixed_S4_TCN_exp_experiment |
| S4_TCN | Polynomial | Mixed_S4_TCN_poly_experiment |
| S4_TCN | Impulse | Mixed_S4_TCN_impulse_experiment |
| S4_TCN | Airy | Mixed_S4_TCN_airy_experiment |
| TCN_S4 | Exponential | Mixed_TCN_S4_exp_experiment |
| TCN_S4 | Polynomial | Mixed_TCN_S4_poly_experiment |
| TCN_S4 | Impulse | Mixed_TCN_S4_impulse_experiment |
| TCN_S4 | Airy | Mixed_TCN_S4_airy_experiment |

## Training Configuration

- **Trainer**: GPU (multi-GPU support)
- **Max epochs**: 300
- **Min epochs**: 10
- **Early stopping**: Monitor `train/loss_epoch`
- **Checkpoint**: Save best model based on `val/loss`
- **GPU allocation**: 0.33 GPU per trial (enables 3 parallel runs per GPU)
- **CPU cores**: 4 per trial

## Results Processing

After training completes, results are automatically:
1. Retrieved from W&B API
2. Aggregated into pandas DataFrames
3. Saved as `.pkl` files in `results/`
4. Visualized in multi-page PDF reports

Expected output files:
- `results/Mixed_S4_S4_{pattern}_experiment_data_clean.pkl`
- `results/Mixed_S4_S4_{pattern}_experiment_S4_report.pdf`
- (Similar for S4_TCN and TCN_S4)

## Research Questions

These experiments aim to answer:

1. **Homogeneous vs Heterogeneous**: Do S4+TCN mixed architectures outperform S4+S4?
2. **Order Effects**: Does S4→TCN perform differently than TCN→S4?
3. **Memory Pattern Sensitivity**: Which architecture handles which decay pattern best?
4. **Capacity Scaling**: How does performance scale with model size across architectures?
5. **S4 vs LSTM/TCN**: How do S4-based mixed models compare to existing LSTM/TCN variants?

## Key Design Decisions

### Why S4_S4, S4_TCN, and TCN_S4?

- **S4_S4**: Establishes baseline for deep S4 (comparable to LSTM_LSTM)
- **S4_TCN**: Tests if S4's global structure + TCN's local refinement is synergistic
- **TCN_S4**: Tests if local feature extraction before structured integration helps

### Why These Memory Patterns?

- **Exponential**: Smooth decay, common in natural processes
- **Polynomial**: Slow decay, tests long-range memory
- **Impulse**: Sharp cutoff, hardest to learn
- **Airy**: Oscillating decay, tests frequency sensitivity

### Why These Model Sizes?

Range [8, 16, 32, 64, 128] allows:
- Fair comparison across architectures
- Analysis of parameter efficiency
- Identification of capacity requirements per pattern

## Compatibility

- **PyTorch**: Compatible with S4 implementation (requires transposed=False)
- **TCN**: Already uses [B, L, C] format (compatible)
- **No code changes required**: All layers work out-of-the-box
- **SequentialModel**: Handles tuple outputs automatically

## Troubleshooting

### CUDA Extension Not Found
If you see warnings about CUDA extensions:
```bash
cd extensions/kernels/
python setup.py install
```
This improves S4 speed and memory efficiency.

### Out of Memory
If GPU OOM occurs:
- Reduce `sweeper.resources.gpu` (e.g., 0.25 instead of 0.33)
- Reduce batch size in `configs/data/lf.yaml`

### Experiment Not Found
Ensure you're in the project root directory:
```bash
cd .
```

## File Structure

```
configs/
├── module/
│   └── lit_module/
│       └── mix_model/
│           ├── S4_S4.yaml
│           ├── S4_TCN.yaml
│           └── TCN_S4.yaml
└── experiment/
    └── Mixed_Models/
        ├── S4_S4/
        │   ├── base.yaml
        │   ├── exp/coarse.yaml
        │   ├── polynomial/coarse.yaml
        │   ├── impulse/coarse.yaml
        │   └── airy/coarse.yaml
        ├── S4_TCN/
        │   ├── base.yaml
        │   ├── exp/coarse.yaml
        │   ├── polynomial/coarse.yaml
        │   ├── impulse/coarse.yaml
        │   └── airy/coarse.yaml
        └── TCN_S4/
            ├── base.yaml
            ├── exp/coarse.yaml
            ├── polynomial/coarse.yaml
            ├── impulse/coarse.yaml
            └── airy/coarse.yaml

scripts/
└── Mixed_Models/
    ├── S4_S4.sh
    ├── S4_TCN.sh
    ├── TCN_S4.sh
    └── run_all_s4_experiments.sh
```

## Citation

If you use these configurations, please cite the original S4 paper:
```
@article{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  journal={ICLR},
  year={2022}
}
```

---

**Created**: 2025-11-05
**Author**: Automated configuration generation
**Status**: Ready for execution
