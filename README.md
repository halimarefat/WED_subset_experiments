<p align="center">
  <img src="/WED_Arch_updated.png" width="650" align="center">
</p>

# WED_subset_experiments

## Environment
- Python 3.10+ recommended.
- Install deps: `pip install -r requirements.txt`

## Running
All configuration is via CLI flags (defaults shown):

```
python train_subset.py \
  --Re R4 \
  --Mconf 3 \
  --model-mode WAE \
  [--wavelet] \
  --train-fraction 1.0 \
  --train-rows None \
  --split-size 0.8 \
  --epochs 5 \
  --learning-rate 0.001 \
  --patience 60 \
  --seed 42
```

### Arguments
- `--Re`: Reynolds number selector (e.g., `R3`, `R4`, `R53`).
- `--Mconf`: Model config key mapping to column headers `M1`–`M5`.
- `--model-mode`: `WAE` or `MLP`.
- `--wavelet`: If set, uses `WaveletLoss`; otherwise MSE.
- `--train-fraction`: Fraction of the available train rows to sample (0–1].
- `--train-rows`: Exact number of train rows to sample (overrides fraction when provided).
- `--split-size`: Portion of sampled data used for training; remainder is validation.
- `--epochs`: Number of epochs.
- `--learning-rate`: Adam learning rate.
- `--patience`: Early-stopping patience on validation loss.
- `--seed`: Seed for reproducibility (NumPy, torch, CUDA if present).

### `Mconf` feature sets
- `1`: velocity + shear (`Ux, Uy, Uz, S1–S6, Cs`)
- `2`: gradients + shear (`G1–G6, S1–S6, Cs`)
- `3`: velocity + subgrid terms (`Ux, Uy, Uz, UUp1–UUp6, Cs`)
- `4`: gradients + subgrid terms (`G1–G6, UUp1–UUp6, Cs`)
- `5`: velocity + gradients + shear + subgrid terms (union of above + `Cs`)

## Outputs
Artifacts land under `runs/<group>/` where `group` is derived from `model-mode` and `Re`.
- `checkpoints/`: best model `*.pt`.
- `logs/`: per-epoch log and JSON history.
- `traced/`: TorchScript traced model for export.

## Data
Download the dataset zip (~0.9 GB) and unpack it at the repo root so that `datasets_csv/` exists alongside `train_subset.py`:
1) Download: https://drive.google.com/file/d/16FZWpGsZ7-_c49XtYwS0JinKBCzSZw1X/view?usp=sharing
2) Place the zip in `WED_subset_experiments/` and extract to produce `datasets_csv/`.

Contents of `datasets_csv/`:
- `original/{train,test}/fieldData_R{3,4,53}_{seen,unseen}.csv`: raw fields.
- `normalized/{train,test}/..._norm.csv`: z-scored fields.
- `coeffs/{train,test}/..._means.csv` and `..._scales.csv`: per-column normalization stats.

Column schema (in order): `t, X, Y, Z, Ux, Uy, Uz, G1-6 (velocity gradients), S1-6 (shear), UUp1-6 (subgrid stresses), Cs (Smagorinsky coefficient)`.
