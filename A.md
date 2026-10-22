## Selected Single-Factor and Combined Ablations

*Mean ± standard deviation over 10 seeds, split 42.*

| Configuration | OA (%) | AA (%) | Kappa (%) |
|---|---:|---:|---:|
| 4-head baseline | 85.55 ± 4.36 | 89.00 ± 2.17 | 81.48 ± 5.23 |
| **8 heads** | **88.42 ± 2.67** | 90.07 ± 1.28 | **84.96 ± 3.21** |
| FFN expansion ×2 | 87.02 ± 1.88 | 89.54 ± 1.77 | 83.22 ± 2.43 |
| FFN residual scale = 2 | 87.00 ± 3.72 | 89.37 ± 1.72 | 83.21 ± 4.42 |
| CNN-module LayerNorm | 88.69 ± 1.83 | 89.38 ± 0.98 | 85.18 ± 2.18 |
| Z-score input normalization | 87.58 ± 2.46 | **90.77 ± 0.87** | 83.99 ± 3.03 |
| Patch size 11 × 11 | 87.64 ± 1.63 | 89.90 ± 2.20 | 83.96 ± 2.06 |
| 8 heads + Z-score | 87.56 ± 2.79 | 90.68 ± 1.24 | 83.97 ± 3.46 |
| 4 heads + LayerNorm + FFN ×2 | 87.19 ± 2.03 | 89.18 ± 1.61 | 83.33 ± 2.58 |
| 4 heads + FFN ×2 + residual scale = 2 | 87.56 ± 1.62 | **90.30 ± 0.85** | 83.88 ± 1.98 |
| 8 heads + patch size 11 × 11 | 87.36 ± 1.99 | 89.22 ± 1.65 | 83.61 ± 2.44 |