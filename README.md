# TCAMIL

**TCAMIL (Two-stage Cluster-aware Aligned Multiple Instance Learning)** is a weakly supervised computational pathology framework for **KRAS mutation prediction** from colorectal cancer (CRC) H&E whole-slide images.

## Overview

TCAMIL is designed to model the histomorphological heterogeneity of colorectal cancer. Instead of directly aggregating patch features, it first constructs a **globally aligned morphology-aware phenotype space** through within-slide clustering and cross-slide alignment, and then performs **cluster-aware hierarchical MIL aggregation** for slide-level prediction.

<p align="center">
  <img src="TCAMIL.png" width="95%">
</p>

## Framework

The main workflow of TCAMIL includes:

1. **WSI tiling**: H&E whole-slide images are tessellated into non-overlapping image tiles.
2. **Dual feature extraction**: AE-CRC extracts morphology-oriented features for clustering, while UNI2 extracts discriminative tile-level features for prediction.
3. **Within-slide local clustering**: morphologically similar tiles within each WSI are grouped into local phenotype units.
4. **Cross-slide global alignment**: local cluster prototypes from different WSIs are aligned into a shared global morphology-aware phenotype vocabulary.
5. **Cluster-aware feature fusion**: globally aligned cluster identities are embedded and fused with tile-level discriminative features.
6. **Hierarchical MIL prediction**: intra-cluster attention aggregates tiles within each phenotype unit, and inter-cluster attention aggregates cluster-level representations for final KRAS mutation prediction.
7. **Interpretability analysis**: attention-dominant phenotypes and high-attention tissue regions are analysed to provide morphology-aware model interpretation.

## Dataset

This project uses two colorectal cancer cohorts:

- **Gansu cohort**: private cohort from Gansu Provincial People's Hospital, 349 patients.
- **SurGen cohort**: publicly available Scottish colorectal cancer cohort, 350 patients.

## Evaluation setting

- **Internal validation**: patient-level cross-validation within the Gansu cohort.
- **External validation**: model trained on the Gansu cohort and tested on the independent SurGen cohort without retraining or fine-tuning.

## Repository structure

```text
.
├── preprocessing/        # WSI preprocessing and tile extraction
├── feature_extraction/   # tile-level feature extraction
├── clustering/           # within-slide local clustering and cross-slide global alignment
├── mil/                  # TCAMIL model and training scripts
├── evaluation/           # evaluation scripts
├── figures/              # framework and manuscript figures
└── README.md
