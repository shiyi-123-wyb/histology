# TCAMIL

**TCAMIL (Two-Stage Clustering Aligned Multiple Instance Learning)** is a weakly supervised computational pathology framework for **KRAS mutation prediction** from colorectal cancer (CRC) H&E whole-slide images.

## Overview

TCAMIL is designed to model the histomorphological heterogeneity of colorectal cancer.  
Instead of directly aggregating patch features, it first builds a **globally aligned phenotype space** through clustering, and then performs **cluster-aware hierarchical MIL aggregation** for slide-level prediction.

## Framework

The main workflow of TCAMIL includes:

1. **Tile extraction** from whole-slide images  
2. **Feature extraction** for each tile  
3. **CRC-specific clustering space construction** using an autoencoder  
4. **Within-slide local clustering** and **cross-slide global alignment**  
5. **Cluster-aware hierarchical MIL** for KRAS mutation prediction  
6. **Interpretability analysis** based on attention and dominant phenotypes  

## Dataset

This project uses two colorectal cancer cohorts:

- **Gansu cohort**: private cohort, 349 patients  
- **SurGen cohort**: public Scottish cohort, 350 patients  

### Evaluation setting

- **Train**: Gansu cohort  
- **Test**: internal Gansu testing and external SurGen testing  

## Repository structure

```text
.
├── preprocessing/        # WSI preprocessing and tile extraction
├── feature_extraction/   # patch-level feature extraction
├── clustering/           # local clustering and global alignment
├── mil/                  # TCAMIL model
├── evaluation/           # evaluation scripts
├── figures/              # figures for README / manuscript
└── README.md
```

## Installation

```bash
git clone https://github.com/your-repo/TCAMIL.git
cd TCAMIL
pip install -r requirements.txt
```

## Usage

### 1. Preprocess WSIs

```bash
python preprocessing/extract_tiles.py
```

### 2. Extract features

```bash
python feature_extraction/extract_features.py
```

### 3. Perform clustering

```bash
python clustering/run_clustering.py
```

### 4. Train TCAMIL

```bash
python mil/train.py
```

### 5. Evaluate

```bash
python evaluation/test.py
```

## Notes

- Please organise your WSI files and metadata before running the pipeline.
- Update the data paths and configuration files according to your local environment.
- The current framework is designed for CRC KRAS mutation prediction with H&E whole-slide images.
