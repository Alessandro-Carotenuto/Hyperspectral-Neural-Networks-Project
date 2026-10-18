# Hyperspectral Neural Networks for Few-Shot Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" height="28" />
  <img src="https://img.shields.io/badge/PyTorch-deep%20learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" height="28" />
  <img src="https://img.shields.io/badge/Dataset-Pavia%20University-0081A7" alt="Pavia University" height="28" />
  <img src="https://img.shields.io/badge/Task-few--shot%20HSI-8A2BE2" alt="Few-shot hyperspectral classification" height="28" />
  <img src="https://img.shields.io/badge/Training-Kaggle%20GPU-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle GPU" height="28" />
</p>

**Few-shot pixel-wise classification of Pavia University HSI with a CNN-Transformer, channel-spatial attention, and sample amplification. The best recorded 8-head configuration achieves 88.42 ± 2.67% OA, 90.07 ± 1.28% AA, and 84.96 ± 3.21% Kappa across 10 runs on a fixed split, with about 30 minutes of training. The project also explores attention-guided spectral-band pruning with STE.**

*Alessandro Carotenuto*

Reconstructed from:

> Chuan Fu, Tianyuan Zhou, Tan Guo, Qikui Zhu, Fulin Luo, and Bo Du, *CNN-Transformer and Channel-Spatial Attention based network for hyperspectral image classification with few samples*, Neural Networks 186 (2025), 107283. [DOI: 10.1016/j.neunet.2025.107283](https://doi.org/10.1016/j.neunet.2025.107283)

The paper proposes CTA-Net for few-sample hyperspectral image classification. This project implements its main components and documents choices that the paper leaves unspecified, including preprocessing, padding, attention details, and sample amplification parameters.

---

## Table of Contents

- [How it works](#how-it-works)
- [Typical results](#typical-results)
- [Ablation highlights](#ablation-highlights)
- [Architecture](#architecture)
- [Training pipeline](#training-pipeline)
- [Project structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Reference](#reference)

---

## How it works

The model predicts the land-cover class of the center pixel in each hyperspectral patch.

```text
Input patch: 15 x 15 pixels, 103 spectral bands
                    |
             Input projection
                    |
          +---------+---------+
          |                   |
      CNN branch        Transformer branch
          |                   |
          +------ concatenate-+
                    |
               CT residual
                    |
          Channel-spatial attention
                    |
        Global average pooling + FC
                    |
             9 class scores
```

The main experiment follows the Pavia University few-shot setup: 10 training pixels and 5 validation pixels per class, with all remaining labeled pixels used for testing. Sample Amplification is applied to training patches only. Validation and test data are left unchanged.

## Typical results

Results are mean ± sample standard deviation over 10 training seeds on the fixed split `42` (10 training and 5 validation samples per class). The baseline uses `15 x 15` patches, 128 feature channels, 150 epochs, Adam with learning rate `0.00008`, and zero weight decay.

| Model | OA | AA | Kappa |
|-------|---:|---:|------:|
| CT + CSA, no sample amplification (4 heads) | 84.10 ± 4.06% | 88.03 ± 2.13% | 79.68 ± 4.77% |
| CTA-Net, with sample amplification (4 heads) | 85.55 ± 4.36% | 89.00 ± 2.17% | 81.48 ± 5.23% |
| CTA-Net, with sample amplification (8 heads) | **88.42 ± 2.67%** | **90.07 ± 1.28%** | **84.96 ± 3.21%** |

At 4 heads, sample amplification improves OA by `1.45` points and Kappa by `1.80` points. The 8-head configuration improves the observed OA by `2.87` points over the 4-head CTA-Net baseline. These are descriptive differences on one fixed split, not significance tests across independent data splits.

## Ablation highlights

The 8-head configuration is the strongest single-factor result for OA and Kappa in the recorded baseline ablations. Other changes did not improve all metrics at once:

| Configuration | Split | OA | AA | Kappa |
|---|---:|---:|---:|---:|
| 8 heads + Z-score input | 42 | 87.56 ± 2.79% | **90.68 ± 1.24%** | 83.97 ± 3.46% |
| Combined settings: 8 heads, 64 channels, patch 11, Z-score, LayerNorm, cosine schedule, and other changes | 42 | 85.09 ± 1.57% | 87.71 ± 1.23% | 80.73 ± 1.94% |
| Same combined settings | 1 | 78.51 ± 5.17% | 86.61 ± 2.12% | 73.12 ± 5.90% |

Combining individually tested settings did not outperform the 8-head baseline and produced different results across the two splits. The full tables and per-class results are in [ablations.md](ablations.md).

## Architecture

The classifier receives patches as `bands × height × width` tensors. A `1 × 1` input projection maps spectral bands to feature channels; the selected classifier then processes those spatial features.

| Configured variant | Components |
|---|---|
| `CNN_CSA` | Multiscale CNN, channel-spatial attention, and classifier |
| `CNN_TRANSFORMER` | Parallel CNN and Transformer branches, then classifier |
| `CNN_TRANSFORMER_CSA` | CNN-Transformer block, channel-spatial attention, and classifier; the complete CTA-Net classifier |
| `CNN_TRANSFORMER_CSA_BAND_PRUNING` | CTA-Net classifier with the supported adaptive spectral-band pruning head |

The classifier variants reuse the same implementation blocks. Sample amplification is a training-data option applied outside the model; it is not a separate neural-network layer.

### CNN branch

Four parallel convolution branches capture spatial features at different receptive fields:

- one `1 x 1` convolution;
- one `3 x 3` convolution;
- two stacked `3 x 3` convolutions;
- three stacked `3 x 3` convolutions.

Branch outputs are concatenated and projected with a `1 x 1` convolution.

### Transformer branch

The Transformer branch follows a Conformer-style block with two feed-forward modules, multi-head self-attention with relative position information, and an internal convolution module. Attention operates over the spatial patch tokens. The CNN and Transformer branches process the same projected feature map in parallel. Their outputs are fused and added to the CT block input through an outer residual connection.

### Channel-spatial attention

The attention block applies channel attention followed by spatial attention. Spatial descriptors include the maximum, minimum, mean, and standard deviation across channels. Each attention stage has a residual connection.

### Adaptive spectral pruning (experimental)

The only supported adaptive pruning architecture is `CNN_TRANSFORMER_CSA_BAND_PRUNING`. It combines pooled output from the existing MHSA branch with one spatial-mean descriptor per input band to score bands. It adds no Transformer and performs no convolution across bands. After a 10-epoch warm-up, STE training uses a global binary mask in the forward pass and soft per-sample gates for the backward gradient. Each pruning budget requires its own training run.

The global bottom-k mask is selected from mean gate scores on the training loader only. For inference, the selected bands are fixed across patches: the normalized image cube is filtered once, and a compact CTA-Net is created with the input projection narrowed to the retained bands. The pruning head is not part of compact-model inference. `PRUNING_FINETUNE_EPOCHS` controls optional fine-tuning of that compact classifier on training data (default `30`; use `0` to skip it). Results with and without compact-model fine-tuning are different protocols.

The five-seed STE sweep uses one fixed train/validation split; the test split was not used. Timing is single-thread CPU at batch size 32, after filtering the normalized cube once:

| Bands removed | Bands kept | Compact validation OA after fine-tuning | Full → compact latency | Ratio |
|---:|---:|---:|---:|---:|
| 70% | 31 | 82.7 ± 3.3% | 2.362 → 2.530 ms/sample | 0.93× |
| 80% | 21 | 83.6 ± 2.0% | 2.210 → 2.126 ms/sample | 1.04× |
| 90% | 10 | 78.7 ± 6.6% | 2.192 → 2.119 ms/sample | 1.03× |

The measured gains are small and noisy; these experiments do not establish a robust inference speedup. A separate one-seed 99% pruning run retained one band, reached 46.7% validation OA after fine-tuning, and measured 1.01× batch-32 speedup. Detailed protocols and historical results are in [ablations.md](ablations.md).

### Sample amplification

The `CTA_NET` mode materializes four training pools for every run: original patches, Gaussian-noise variants outside a protected central `3 x 3` region, randomly rotated patches, and same-class direct-sum patches. With 10 original patches per class, this implementation produces 39 patches per class, 351 total. The augmentation seed is tied to the run's training seed for reproducibility.

## Training pipeline

1. Load Pavia University and its ground-truth labels.
2. Create a reproducible few-shot split with 10 training and 5 validation pixels per class.
3. Normalize each spectral band using min-max scaling across the scene, then reflect-pad the cube and extract `15 x 15` patches.
4. Select an architecture and materialize the training data, optionally using CTA-Net sample amplification.
5. Train with Adam and save the checkpoint with the lowest validation loss.
6. Evaluate per-class accuracy, Overall Accuracy (OA), Average Accuracy (AA), Cohen's Kappa, and the confusion matrix.
7. Save run configuration and test metrics as JSON. In multi-seed mode, also save mean and sample standard deviation across runs.

## Project structure

```text
.
├── data.py                              # Splits, normalization, patch datasets, SA
├── evaluation.py                        # Metrics, reports, classification maps
├── models.py                            # CNN, Transformer, CSA architectures
├── plotting.py                          # Result and training visualizations
├── training.py                          # Training loop, seeds, checkpoints
├── utils.py                             # Enumerations and shared configuration types
├── ablations.md                          # Detailed accuracy, pruning, and timing results
├── hsi_classification.ipynb              # Main experiment and analysis notebook
├── docs/
│   └── research/                         # Method and replication notes
└── tests/                                # Focused data and configuration tests
```

The Pavia University `.mat` files are not included. Place `PaviaU.mat` and `PaviaU_gt.mat` in `data/raw/` for local runs, or attach them as notebook inputs in Kaggle.

## Configuration

The notebook keeps the experiment configuration together near its beginning. Important options include:

| Setting | Current value | Description |
|----------|---------------|-------------|
| `PATCH_SIZE` | `15` | Spatial patch width and height |
| `TRAIN_SAMPLES_PER_CLASS` | `10` | Few-shot training pixels per class |
| `VALIDATION_SAMPLES_PER_CLASS` | `5` | Validation pixels per class |
| `TRAINING_SEED_MODE` | `ONE_FIXED_TRAIN_SEED` | One fixed seed, a fixed reusable seed list, or newly randomized multi-seed runs |
| `NUM_SEEDS` | `10` | Runs in multi-training-seed mode |
| `TRAINING_SEEDS_FIXED` | `[]` | Paste the printed seed list here when using `MULTI_FIXED_TRAIN_SEED` |
| `ARCHITECTURE` | `CNN_TRANSFORMER_CSA` | Select CNN, CT, CSA, or CT + CSA |
| `TRANSFORMER_HEADS` | `4` | Notebook default; the best recorded baseline ablation uses `8` heads |
| `TRANSFORMER_FFN_EXPANSION` | `4` | Feed-forward hidden-width multiplier |
| `SAMPLE_AMPLIFICATION_MODE` | `CTA_NET` | Disable SA or use the CTA-Net augmentation |
| `INPUT_NORMALIZATION` | `MIN_MAX` | Per-band min-max or z-score normalization |
| `PRUNING_GATE_SLOPE` | `20.0` | Fixed sigmoid slope for adaptive spectral gates |
| `PRUNING_SPARSITY_WEIGHT` | `0.001` | Loss weight encouraging low gate values |
| `PRUNING_PERCENTAGE` | `5.0` | Bottom percentage of mean training gates to remove; rounded to the nearest whole number of bands (5 of 103 at the default) |
| `PRUNING_FINETUNE_EPOCHS` | `30` | Compact-model fine-tuning epochs; set to `0` to skip fine-tuning |
| `WEIGHT_DECAY` | `0.0` | Adam weight decay |

The SA values that are not specified by the paper are documented in [the sample amplification note](docs/research/cta_sample_amplification.md). Transformer reconstruction choices are summarized in [the Conformer analysis](docs/research/hsi_conformer_architectures.md).

## Usage

Open `hsi_classification.ipynb`, configure the dataset paths and experiment settings, then run the cells in order. Training always uses the configured number of epochs in `EPOCHS`.

For each full run, the notebook writes checkpoints under `outputs/checkpoints/` and evaluation reports under `outputs/results/`. Each report includes the configuration, split and training seeds, sample amplification metadata, validation-selected checkpoint, and test metrics.

The reusable components can also be imported from `data.py`, `models.py`, `training.py`, and `evaluation.py` for custom experiments.

## Reproducibility and limitations

- The current ten-run result uses one fixed few-shot split and ten training seeds. A full ten-split study is planned separately.
- The paper does not specify every Transformer, preprocessing, or sample amplification parameter. Those implementation choices are recorded in the research notes and run reports.
- Per-band min-max normalization uses the full image cube. It does not use labels, but it uses feature statistics from the complete scene.
- Random pixel splits allow neighboring patches to overlap spatially. Spatially separated splits are available as a separate robustness experiment.
- The reported local mean OA is below the paper's absolute OA. The observed OA improvement from sample amplification is close to the paper's ablation gain.

## Reference

```bibtex
@article{fu2025ctanet,
  title   = {CNN-Transformer and Channel-Spatial Attention based network for hyperspectral image classification with few samples},
  author  = {Fu, Chuan and Zhou, Tianyuan and Guo, Tan and Zhu, Qikui and Luo, Fulin and Du, Bo},
  journal = {Neural Networks},
  volume  = {186},
  pages   = {107283},
  year    = {2025},
  doi     = {10.1016/j.neunet.2025.107283}
}
```
