# Hyperspectral Neural Networks for Few-Shot Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" height="28" />
  <img src="https://img.shields.io/badge/PyTorch-deep%20learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" height="28" />
  <img src="https://img.shields.io/badge/Dataset-Pavia%20University-0081A7" alt="Pavia University" height="28" />
  <img src="https://img.shields.io/badge/Task-few--shot%20HSI-8A2BE2" alt="Few-shot hyperspectral classification" height="28" />
  <img src="https://img.shields.io/badge/Training-Kaggle%20GPU-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle GPU" height="28" />
</p>

**Pixel-wise classification of hyperspectral imagery with only 10 labeled training pixels per class. The project builds CNN, CNN-Transformer, channel-spatial attention, and sample amplification components, then evaluates the complete CTA-Net pipeline on Pavia University.**

*Alessandro Carotenuto*

Reconstructed from:

> Chuan Fu, Tianyuan Zhou, Tan Guo, Qikui Zhu, Fulin Luo, and Bo Du, *CNN-Transformer and Channel-Spatial Attention based network for hyperspectral image classification with few samples*, Neural Networks 186 (2025), 107283. [DOI: 10.1016/j.neunet.2025.107283](https://doi.org/10.1016/j.neunet.2025.107283)

The paper proposes CTA-Net for few-sample hyperspectral image classification. This project implements its main components and documents choices that the paper leaves unspecified, including preprocessing, padding, attention details, and sample amplification parameters.

---

## Table of Contents

- [How it works](#how-it-works)
- [Typical results](#typical-results)
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

The current ten-run experiment holds the data split fixed at seed `42` and varies the training seed. It uses 10 training samples and 5 validation samples per class, `15 x 15` patches, 128 feature channels, 150 epochs, Adam with learning rate `0.00008`, and zero weight decay.

| Model | OA | AA | Kappa |
|-------|---:|---:|------:|
| CT + CSA, no sample amplification | 84.06 ± 2.28% | 88.88 ± 1.03% | 79.76 ± 2.59% |
| CTA-Net, with sample amplification | **85.47 ± 3.55%** | **88.88 ± 2.23%** | **81.37 ± 4.19%** |

The measured sample amplification gain is `+1.41` percentage points in OA and `+1.61` in Kappa. The paper reports a `+1.23` point OA gain for its corresponding ablation on Pavia University. The effect size is similar, although the absolute accuracy remains lower. These results are from repeated training seeds on one fixed split, not from ten independently sampled data splits.

## Architecture

### CNN branch

Four parallel convolution branches capture spatial features at different receptive fields:

- one `1 x 1` convolution;
- one `3 x 3` convolution;
- two stacked `3 x 3` convolutions;
- three stacked `3 x 3` convolutions.

Branch outputs are concatenated and projected with a `1 x 1` convolution.

### Transformer branch

The Transformer branch follows a Conformer-style block with two feed-forward modules, multi-head self-attention with relative position information, and an internal convolution module. The CNN and Transformer branches process the same projected feature map in parallel. Their outputs are fused and added to the CT block input through an outer residual connection.

### Channel-spatial attention

The attention block applies channel attention followed by spatial attention. Spatial descriptors include the maximum, minimum, mean, and standard deviation across channels. Each attention stage has a residual connection.

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
├── notebooks/
│   └── hsi_classification.ipynb          # Main experiment and analysis notebook
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
| `SAMPLE_AMPLIFICATION_MODE` | `CTA_NET` | Disable SA or use the CTA-Net augmentation |
| `INPUT_NORMALIZATION` | `MIN_MAX` | Per-band min-max or z-score normalization |
| `WEIGHT_DECAY` | `0.0` | Adam weight decay |

The SA values that are not specified by the paper are documented in [the sample amplification note](docs/research/cta_sample_amplification.md). Transformer reconstruction choices are summarized in [the Conformer analysis](docs/research/hsi_conformer_architectures.md).

## Usage

Open `notebooks/hsi_classification.ipynb`, configure the dataset paths and experiment settings, then run the cells in order. Set `RUN_FULL_TRAINING = False` for a short smoke run. Set it to `True` for the configured 150-epoch experiment.

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
