# TCAMIL

Official implementation of **TCAMIL**: cluster-aware hierarchical MIL for gigapixel pathological image classification.

This repository implements a **3-stage pipeline**:

1) **Local clustering (per WSI)** → generates `cluster_assignments.csv`  
2) **Fold-wise global clustering (train+val fit, test predict)** → generates  
   `coord_to_global_cluster_trainval.pkl` and `coord_to_global_cluster_test.pkl`  
3) **TCAMIL training/testing (final model: Only-Embedding)** → hierarchical MIL with  
   intra-cluster attention + inter-cluster attention + cluster embedding fusion

---

## Quick Start (copy & run)

> Replace the placeholders:
> - `<patch_root>`: patch image root (per-WSI folders)
> - `<cluster_root>`: output root for clustering results
> - `<feature_dir>`: H5 features directory (`<WSI_ID>.h5`)
> - `<label_csv>`: labels CSV (`ID,label`)
> - `<split_dir>`: splits directory (`splits_k.csv`)

```bash
############################
# 1) Environment
############################
conda create -n tcamil python=3.10 -y
conda activate tcamil
pip install -r requirements.txt

# (Optional) GPU acceleration for PCA / clustering
# If not installed, the code automatically falls back to CPU.
# Example only (adjust to your CUDA environment):
# pip install cupy-cuda12x
# pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com


############################
# 2) Data format (reference)
############################
# patch folders:
# <patch_root>/
#   <WSI_001>/ x123_y456.png ...
#   <WSI_002>/ ...
#
# features:
# <feature_dir>/
#   <WSI_ID>.h5   (contains: features (N,D), coords (N,2))
#
# labels:
# <label_csv> format:
# ID,label
# 001,0
# 002,1
#
# splits:
# <split_dir>/splits_0.csv, splits_1.csv, ...
# each has columns: train,val,test


############################
# 3) Step 1: Local clustering (per WSI)
############################
# Script: run_clustering.py
# Output: <cluster_root>/<WSI_ID>/cluster_assignments.csv

# NOTE: run_clustering.py currently uses hardcoded paths inside main().
# You MUST edit these lines in run_clustering.py before running:
#   args.input_path  = "<patch_root>"
#   args.output_path = "<cluster_root>"
# and other options if needed (gpu_ids, dim_reduce, etc.)

python run_clustering.py


############################
# 4) Step 2 + 3: Global clustering + TCAMIL training/testing
############################
# Script: baseline_only_embedding.py
# It will automatically run fold-wise global clustering (Step 2) before MIL training/testing (Step 3).

# ---- Train (fold k=0) ----
python baseline_only_embedding.py \
  --phase train \
  --dataset CustomDataset \
  --k 0 \
  --feature_dir <feature_dir> \
  --label_csv <label_csv> \
  --split_dir <split_dir> \
  --cluster_root <cluster_root> \
  --num_clusters 18 \
  --feature_dim 1536 \
  --n_epochs 50 \
  --lr 0.0005 \
  --wd 0.0001 \
  --label_smoothing 0.05 \
  --slide_dropout 0.3 \
  --max_patches_per_cluster 300 \
  --cluster_emb_dim 8

# ---- Test (fold k=0) ----
python baseline_only_embedding.py \
  --phase test \
  --dataset CustomDataset \
  --k 0 \
  --feature_dir <feature_dir> \
  --label_csv <label_csv> \
  --split_dir <split_dir> \
  --cluster_root <cluster_root> \
  --num_clusters 18 \
  --feature_dim 1536 \
  --cluster_emb_dim 8

# ---- Run all folds (example: 10-fold) ----
for k in 0 1 2 3 4 5 6 7 8 9; do
  python baseline_only_embedding.py \
    --phase train \
    --dataset CustomDataset \
    --k ${k} \
    --feature_dir <feature_dir> \
    --label_csv <label_csv> \
    --split_dir <split_dir> \
    --cluster_root <cluster_root> \
    --num_clusters 18 \
    --feature_dim 1536 \
    --n_epochs 50 \
    --lr 0.0005 \
    --wd 0.0001 \
    --label_smoothing 0.05 \
    --slide_dropout 0.3 \
    --max_patches_per_cluster 300 \
    --cluster_emb_dim 8
done
