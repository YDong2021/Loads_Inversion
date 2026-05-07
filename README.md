# Impact Load Localization & Inversion (ResNet + Mamba)

A two-stage deep learning framework that takes a 3-channel acceleration
response signal and produces:

1. **Impact location classification** (class id in `[0, 49]`, ResNet1D)
2. **Force–time curve regression** (Mamba / Selective SSM)

## Environment Setup

```powershell
conda create -n loads_inv python=3.10 -y
conda activate loads_inv
pip install -r requirements.txt
# For a CUDA build of PyTorch, install the matching wheel from pytorch.org
```

## Data

> ⚠️ **Sensitive data — NOT shipped with the repo.**
> The entire `data/` directory is listed in [.gitignore](.gitignore) and will
> never be committed. After cloning, place the two files below into `data/`
> manually before running any script.

- `data/diverse_impact_signals.xlsx`: 200 trapezoidal load signals
  (`2500 × 201`, one `Time` column + `Signal_001`…`Signal_200`)
- `data/batch_response_results.h5`: 30,000 responses with shape
  `(2498, 30000)`, carrying the indices
  `force_signal_indices / frf_indices / node_ids / custom_mapping_ids`

Each sample is one `(force_idx, node_id)` pair stacked over the 3 FRF
channels, giving a `(3, 2498)` response tensor.

In addition, common artefact types (`*.h5 / *.hdf5 / *.npy / *.npz /
*.xlsx`) as well as `checkpoints/`, `logs/`, and `outputs/` are also
ignored to prevent accidentally committing weights or evaluation dumps.

## Directory Layout

```
configs/    # YAML hyper-parameters (default + per-experiment overrides)
data/       # raw data files + Dataset / transforms
models/     # layers → blocks → networks → losses
engine/     # trainer / evaluator
utils/      # logger / checkpoint / metrics / seed
scripts/    # one-off helpers (data exploration, waveform plotting)
```

## Training Pipeline

```powershell
# 1. Train the position classifier
python train_classifier.py --config configs/classifier_6_8_10.yaml

# 2. Train the shape regressor (teacher forcing with ground-truth pos_id)
python train_regressor.py --config configs/regressor_mamba8.yaml

# 3. End-to-end evaluation on the test split
python eval.py --classifier-ckpt checkpoints/classifier_6_8_10/best.pth \
               --regressor-ckpt  checkpoints/regressor_mamba8/best.pth \
               --out-dir outputs/eval_e2e

# 4. Inference on a single / batch response tensor
python infer.py --classifier-ckpt checkpoints/classifier_6_8_10/best.pth \
                --regressor-ckpt  checkpoints/regressor_mamba8/best.pth \
                --input path/to/response.npy \
                --out-dir outputs/infer_single
```

## Model Variants

| Module              | Variants                                                              | Notes                                              |
| ------------------- | --------------------------------------------------------------------- | -------------------------------------------------- |
| Position classifier | `resnet1d_6_8_10` (default), `resnet1d_4_6_8`, `resnet1d_8_8_8`, `resnet1d_8_10_12` | conv count per `[stage1, stage2, stage3]`          |
| Mamba regressor     | `mamba_6`, `mamba_8` (default), `mamba_10`                            | number of stacked Mamba blocks                     |

## Key Design Decisions

- **Load resampling**: force curves are linearly resampled `2500 → 2498`
  to align with the response length.
- **Split**: `train / val / test = 0.70 / 0.15 / 0.15` over `(force_idx, node_id)` pairs.
- **Global peak normalization**: the scalar `max|value|` is fit on the
  training subset only and reused at inference time.
- **Pure-PyTorch Selective SSM**: implemented with an explicit `O(L)`
  time-step scan (no CUDA kernel required).
- **Teacher forcing**: the regressor is trained with the *true* `pos_id`;
  only `infer.py` / `eval.py` feed the classifier's prediction into the
  Fourier positional encoding.
- **Combined impact loss**:
  `L = 0.8·L_MSE + 0.4·L_grad + 1.0·L_stage + 0.8·L_peak`,
  with stage weights `w_accel = 1.3`, `w_inertia = 1.0`, `w_decay = 0.2`.

## Helper Scripts

```powershell
# Raw-data sanity check (force Excel + response HDF5)
python -m scripts.explore_data

# Re-plot waveforms from a preds.npz produced by eval.py
python -m scripts.plot_waveforms --preds outputs/eval_e2e/preds.npz \
                                 --out-dir outputs/eval_e2e/waveforms_extra \
                                 --mode grid --rows 4 --cols 6
```
