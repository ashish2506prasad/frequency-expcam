# EXP-CAM

Experiments for EXP-CAM: sequential loss ablation, mask-budget minimality, and baseline comparisons on the `trainx/` ImageNet subset.

![Baseline comparison (ResNet18, heatmap grid)](assets/comparison_heatmap_grid.png)

## Setup

```bash
conda activate vae
pip install PyYAML
cd /home/medal/21d180009

# Build or refresh the cached eval set (all trainx classes with ≥10 images)
python -m expcam eval-set --where local --force --out expcam_runs_wo_ce
```

---

## Section 0 — polar FFT (generative + attribution)

WaveX-style **rfft2** polar bins (default 16 radial × 16 angular). A U-Net on RGB maps to one shared sigmoid per bin; `e = IFFT(m ⊙ FFT(x))`.

**Best-mask metrics** (both logged every step; `--fft-metric` picks the checkpoint and the greedy/LPF comparison):

- `p_energy` = fraction of total FFT energy in the explanation
- `HM_conf` = HM(`1 − p_energy`, `conf(e)`)
- `HM_rob` = HM(`1 − p_energy`, `conf_rob`) where dropped frequencies are replaced with white noise before IFFT

**Minimality:** `L_energy` vs `L_area` (polar-bin mean), analogous to pixel EXP-CAM `L_area`. Weights in `expcam/configs/fft.yaml`.

**LPF vs greedy:** for radii `r_i` linearly spaced from 0 to 1 (`n_lpf_radii`, default 50), take the circular low-pass energy `e_i`, then greedily fill polar bins by U-Net score until the same energy. Plot `Δconf = conf(x) − conf(filtered)` for both.

**Attribution-only:** circular low-pass integrated gradients (WaveX Fourier IG).

Eval set: **10 images per class** for every `trainx/` synset with at least 10 files (`n01749939` / green mamba is skipped). Outputs: `section0_fft/`.

```bash
# Train + HM curves + step grids + WaveX FFT figures + LPF/greedy + IG
python -m expcam 0 --config expcam/configs/fft.yaml \
  --where local --out freq_expcam_fft --backbones resnet18 vit_b_16

# Checkpoint / greedy comparison on robustness confidence
python -m expcam 0 --fft-metric energy_rob --config expcam/configs/fft.yaml \
  --where local --out freq_expcam_fft --backbones resnet18

# Area-only vs energy-only (compare to pixel EXP-CAM L_area)
python -m expcam 0 --lam-energy 0 --lam-area 15 --where local --out freq_expcam_fft_area
python -m expcam 0 --lam-energy 10 --lam-area 0 --where local --out freq_expcam_fft_energy

# Figures only
python -m expcam 0 --figures-only --where local --out freq_expcam_fft --backbones resnet18 vit_b_16
```

**Outputs:** `.../section0_fft/{backbone}/ef1_full/{id}/` (`best_ef1/`, `snapshots/`, `attribution_ig/`) and `figures_{backbone}/`.

Sections 1–3 are unchanged pixel EXP-CAM.

---

## Section 1 — Sequential loss ablation

Trains cumulative loss stages (`L_act` → … → `Full`) on the eval set. Outputs bar charts and qualitative mask grids per image.

```bash
# Train + figures
python -m expcam 1 --where local --out expcam_runs_wo_ce --backbones resnet18 vit_b_16

# Figures only (from saved metrics.json)
python -m expcam 1 --figures-only --where local --out expcam_runs_wo_ce --backbones resnet18 vit_b_16
```

**Outputs:** `expcam_runs_wo_ce/section1/figures_{backbone}/individual/SEQ_*.png`, `seq_qualitative/`, `FigA_SEQ_qualitative_*.png`

---

## Section 2 — Minimality (knapsack on Section 0 masks)

Post-hoc top-k knapsack sweep on Section 0 best `mask_prob` at multiple budget levels. No retraining.

```bash
# Knapsack metrics + plots + mask grids
python -m expcam 2 --from-section0-knapsack --section0-source section0_new_score \
  --where local --out expcam_runs_wo_ce --backbones resnet18 vit_b_16

# Figures only
python -m expcam 2 --from-section0-knapsack --figures-only \
  --section0-source section0_new_score \
  --where local --out expcam_runs_wo_ce --backbones resnet18 vit_b_16
```

**Outputs:** `expcam_runs_wo_ce/section2/figures_{backbone}/*.png`, `mask_grids/`

---

## Section 3 — Baseline comparison

Train EXP-CAM and compare against CAM baselines at size-matched sparsity. Per-image fair and heatmap comparison panels.

```bash
# Train EXP-CAM + run baselines + aggregate figures
python -m expcam 3 --where local --out expcam_runs_wo_ce --backbones resnet18 vit_b_16

# Per-image comparison grids only
python -m expcam 3 --figures-only --comparisons-only \
  --where local --out expcam_runs_wo_ce --backbones resnet18 vit_b_16

# Custom heatmap grid (example)
python -m expcam 3 --figures-only --comparisons-only \
  --grid-ids 006 019 030 054 061 079 \
  --where local --out expcam_runs_wo_ce --backbones resnet18
```

**Outputs:** `expcam_runs_wo_ce/section3/figures_{backbone}/comparisons_fair/`, `comparisons_heatmap/`, `FigC*.png`

---

## Module layout

| Command | Module |
|---------|--------|
| `python -m expcam 0` | `ef1.py` |
| `python -m expcam 1` | `ablation.py` |
| `python -m expcam 2` | `minimality.py` |
| `python -m expcam 3` | `comparisons.py` |

Run directories stay under `expcam_runs_wo_ce/section1/`, `section2/`, `section3/` for compatibility with existing results.
