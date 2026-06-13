# JPEG RDH — Block Sorting Optimization & Dynamic Iterative Histogram Modification

MATLAB implementation of:

> F. Li, Q. Wang, H. Cheng, X. Zhang, C. Qin,
> **"JPEG Reversible Data Hiding via Block Sorting Optimization and Dynamic
> Iterative Histogram Modification"**,
> *IEEE Transactions on Multimedia*, vol. 27, pp. 3729–3743, 2025.
> DOI: 10.1109/TMM.2025.3535320

---

## Overview

A reversible data hiding (RDH) scheme for JPEG images that embeds secret data
into quantized DCT coefficients and **perfectly restores both the secret data
and the original JPEG image**. It balances embedding capacity, visual quality,
and JPEG file-size increment via three contributions:

1. **Block sorting optimization** (Eqs. 11–12) — a smoothness score combining
   the number of zero AC coefficients with the quantization-table values of the
   non-zero ones, so smoother blocks are embedded first (lower distortion).
2. **Optimal frequency-band selection** (Eqs. 13–15) — rank the 63 AC
   "frequency bands" by unit distortion and pick the minimum-distortion set
   `F*` that still meets the target capacity `P`.
3. **Dynamic iterative histogram modification** (Algorithm 1, Eqs. 16–18) —
   search the embedding configuration that minimises image MSE under the
   capacity constraint.

### Pipeline (Fig. 2)

```
            ┌──────────── Content owner / data hider ────────────┐
JPEG decode │  quantized DCT coefficients                        │
  ─────────►│   → block smoothness sort        (Eqs. 11-12)      │
            │   → optimal frequency band F*    (Eqs. 13-15)      │──► marked
            │   → histogram-shift embed        (Eqs. 1-2)        │    image
            └────────────────────────────────────────────────────┘
            ┌──────────────────── Receiver ──────────────────────┐
 marked ───►│  re-sort (smoothness invariant)                    │──► secret +
  image     │   → extract secret               (Eqs. 3-4)        │    perfect
            │   → recover coefficients → inverse JPEG            │    original
            └────────────────────────────────────────────────────┘
```

The smoothness score is **invariant under embedding** (zeros stay zero,
non-zeros stay non-zero), so the receiver reproduces the embedding block order
exactly — the key to reversibility of the sorting step.

---

## Repository Structure

```
jpeg-rdh-blocksort-matlab/
├── main_demo.m              # End-to-end demo with figures
├── run_all_tests.m          # Unit tests T1–T8
├── run_experiments.m        # Reproduces Section IV analyses (Figs 10-12 etc.)
├── validate_algorithm.py    # Python pre-validation of core primitives (V1–V4)
├── validate_pipeline.py     # Python pre-validation of the full pipeline (V5–V6)
├── src/
│   ├── jpeg_qtable.m         # JPEG luminance quantization table for a QF
│   ├── zigzag_index.m        # 64×2 zigzag scan positions (DC + F_1..F_63)
│   ├── dct_matrix.m          # 8×8 orthonormal DCT matrix (no toolbox)
│   ├── jpeg_forward.m        # image → quantized DCT coefficients
│   ├── jpeg_inverse.m        # quantized coefficients → image (for PSNR/SSIM)
│   ├── smoothness_scores.m   # block smoothness score          (Eqs. 11-12)
│   ├── block_order.m         # descending-smoothness block order
│   ├── select_bands.m        # optimal frequency band set F*   (Eqs. 13-15)
│   ├── hs_embed_coeff.m      # 1-D histogram-shifting embed     (Eqs. 1-2)
│   ├── hs_extract_coeff.m    # 1-D histogram-shifting extract   (Eqs. 3-4)
│   ├── embed_data.m          # main embedding (Algorithm 1)
│   ├── extract_data.m        # extraction + coefficient recovery
│   ├── embed_data_optimized.m# dynamic min-MSE band-budget search (Eqs. 16-18)
│   └── estimate_jpeg_bits.m  # JPEG AC entropy-size proxy (File Size Increment)
└── utils/
    ├── compute_quality.m     # PSNR (Eq. 19) and SSIM (Eq. 20)
    └── make_test_image.m     # synthetic smooth/texture/mixed test images
```

---

## Quick Start

```matlab
addpath(genpath('src'));
addpath(genpath('utils'));

img = imread('your_grayscale_image.png');   % dims multiple of 8
QF  = 70;

% Owner / data hider
[coeffs, T_zz, dims] = jpeg_forward(img, QF);
secret = randi([0,1], 1, 5000);
[marked, aux] = embed_data(coeffs, T_zz, secret);
marked_img = jpeg_inverse(marked, T_zz, dims);

% Receiver
[secret_out, rec_coeffs] = extract_data(marked, T_zz, aux);

isequal(secret_out, secret)        % 1 — data bit-exact
isequal(rec_coeffs, coeffs)        % 1 — image perfectly restored
compute_quality('psnr', jpeg_inverse(coeffs,T_zz,dims), marked_img)
```

Run the demo, tests, and experiments:
```matlab
main_demo
run_all_tests
run_experiments
```

---

## Algorithm Details

### Histogram shifting (Eqs. 1–4)
Embeddable coefficients are the `±1` AC coefficients; shiftable are `|coef| ≥ 2`.
```
Embed :  |E|=1 → E + sign(E)·t   (±1 stays for t=0, expands to ±2 for t=1)
         |E|>1 → E + sign(E)      (shift outward, vacating ±2)
Extract: 1≤|Ê|≤2 → Ê' = sign(Ê),  bit = 0 if |Ê|=1 else 1
         |Ê|≥3   → Ê' = Ê − sign(Ê)
```

### Block smoothness score (Eqs. 11–12)
```
τ_k = number of zero AC coefficients in block k
E_k = Σ_{s,w} [Q_k(s,w) ≠ 0] · T(s,w)²
S_k = τ_k + τ_k / E_k          (higher = smoother = embedded earlier)
```

### Unit band distortion (Eq. 13)
```
D(F_i) = T_i² · J_i / C_i + ½ · T_i²
   C_i = #{|coef| = 1} in band i,  J_i = #{|coef| > 1}
```
Bands are added in ascending distortion until capacity ≥ `P` (Eq. 18).

---

## Validation Without MATLAB

This machine has no MATLAB/Octave, so the algorithm was first validated in
Python and only then ported:

```
$ python validate_algorithm.py
V1 PASS: 1D histogram-shifting reversible for all coefficients
V2 PASS: smoothness score invariant under embedding (Eqs 11-12)
V3 PASS: unit band distortion ranking (63 bands ranked ...)
V4 PASS: 2D corner-expansion mapping reversible (lattice 14x14)

$ python validate_pipeline.py        # needs numpy + scipy
V5 PASS qf=50/70/90: payload bits, coeffs+data fully recovered
V6 PASS: dynamic band-count selection reversible at 20/40/60% load
FULL PIPELINE VALIDATED — safe to port to MATLAB.
```

The MATLAB `run_all_tests.m` mirrors these checks (T1–T8).

### Implementation note
The 1-D histogram-shifting engine (Eqs. 1–4) is the workhorse and is exactly
as specified in the paper. The paper's two-dimensional coefficient-pair mapping
(Eqs. 5–8) is provided in `validate_algorithm.py` as a provably-reversible
**corner-expansion** form (corner `(1,1)` carries 1.5 bits on average; both
axes shift by one to vacate). The full Type-A/B/C/D arm expansion with cascading
vacation is a faithful extension left as a documented enhancement; the shipped
MATLAB pipeline uses the 1-D engine with block sorting + frequency selection +
dynamic min-MSE band selection, which is fully reversible end-to-end.

---

## Requirements
- MATLAB R2018b+ (for `sgtitle`/`yyaxis` in the demo/experiment plots; the core
  `src/` and `run_all_tests.m` need only base MATLAB — no toolboxes).
- Python 3 with `numpy` (and `scipy` for `validate_pipeline.py`) to re-run the
  pre-validation.

---

## Reference

```bibtex
@article{li2025jpegrdh,
  author  = {Li, Fengyong and Wang, Qiankuan and Cheng, Hang and
             Zhang, Xinpeng and Qin, Chuan},
  title   = {JPEG Reversible Data Hiding via Block Sorting Optimization and
             Dynamic Iterative Histogram Modification},
  journal = {IEEE Transactions on Multimedia},
  year    = {2025},
  volume  = {27},
  pages   = {3729--3743},
  doi     = {10.1109/TMM.2025.3535320}
}
```
