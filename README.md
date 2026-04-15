# TCAMIL

Official implementation of **TCAMIL** (cluster-aware hierarchical MIL for gigapixel pathological image classification).

This repo implements a **3-stage pipeline**:

1) **Local clustering (per WSI)** → generates `cluster_assignments.csv` inside each WSI output folder  
2) **Fold-wise global clustering (train+val fit, test predict)** → generates:
   - `coord_to_global_cluster_trainval.pkl`
   - `coord_to_global_cluster_test.pkl`
3) **TCAMIL training/testing (final model: Only-Embedding)** → hierarchical MIL with:
   - cluster-aware grouping
   - intra-cluster attention pooling
   - inter-cluster attention pooling
   - **cluster embedding fusion enabled** (final setting)

---

## 1) Environment

### 1.1 Create environment
```bash
conda create -n tcamil python=3.10 -y
conda activate tcamil
