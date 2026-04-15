# ===== baseline_only_embedding.py =====
import argparse
import csv
import logging
import os
import random
import numpy as np
import torch

from DMIN_only_embedding import DMINMIL
from clustering_pipeline import run_fold_clustering


def seed_all(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fmt = "%(levelname)s - %(asctime)s - %(filename)s:%(lineno)d - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]
    )


def build_exp_dir(args) -> str:
    dataset_abbr = {
        "Camelyon16": "C",
        "TCGA-NSCLC": "N",
        "TCGA-BRCA": "B",
        "TCGA-RCC": "R",
        "CustomDataset": "Custom",
    }
    pretrain_abbr = {"ResNet50_ImageNet": "Res50"}

    exp_dir = os.path.join(
        "./experiments",
        f"{dataset_abbr[args.dataset]}{args.fold}",
        pretrain_abbr.get(args.pretrain, args.pretrain),
        f"label_frac={args.label_frac}",
        f"ls={args.label_smoothing}_sd={args.slide_dropout}_mppc={args.max_patches_per_cluster}"
        f"_nc={args.num_clusters}_lr={args.lr}",
        "ablation=only_embedding_emb=1_hist=0"
    )
    return exp_dir


def init_dataset_paths(args):
    # 只保留你的 CustomDataset，其他数据集路径先占位（避免 repo 里写死你本地路径）
    if args.dataset == "CustomDataset":
        args.n_classes = 2
        args.subtyping = False
        args.k_sample = 16
        args.n_groups = 4
        args.feature_dir = args.feature_dir or "/path/to/features/"
        args.label_csv = args.label_csv or "/path/to/labels.csv"
        args.split_dir = args.split_dir or "/path/to/splits/"
    else:
        raise NotImplementedError(
            "建议在 GitHub 里不要硬编码你的私有路径；"
            "如需开放其它数据集，把路径通过 CLI 传入。"
        )


def parse_args():
    p = argparse.ArgumentParser("Only-Embedding ablation (final)")

    # run
    p.add_argument("--phase", type=str, default="train", choices=["train", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=10)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")

    # data/model
    p.add_argument("--dataset", type=str, default="CustomDataset",
                   choices=["Camelyon16", "TCGA-NSCLC", "TCGA-BRCA", "TCGA-RCC", "CustomDataset"])
    p.add_argument("--pretrain", type=str, default="ResNet50_ImageNet")
    p.add_argument("--feature_dim", type=int, default=1536)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-4)

    # regularization
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--slide_dropout", type=float, default=0.3)
    p.add_argument("--max_patches_per_cluster", type=int, default=300)

    # clustering
    p.add_argument("--num_clusters", type=int, default=18)
    p.add_argument("--global_pca_dim", type=int, default=32)
    p.add_argument("--p_landmarks", type=int, default=128)
    p.add_argument("--r_neighbors", type=int, default=5)
    p.add_argument("--lsc_mode", type=str, default="random", choices=["random", "kmeans"])

    p.add_argument("--cluster_root", type=str, default="/path/to/local-cluster-output/")

    # paths (建议用 CLI 注入，不在 repo 写死私有目录)
    p.add_argument("--feature_dir", type=str, default="")
    p.add_argument("--label_csv", type=str, default="")
    p.add_argument("--split_dir", type=str, default="")
    p.add_argument("--label_frac", type=float, default=1.0)

    # embedding
    p.add_argument("--cluster_emb_dim", type=int, default=8)

    # optional override coord pkl
    p.add_argument("--coord_pkl_path", type=str, default="")

    args = p.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_dataset_paths(args)

    # ✅ Only-Embedding 强制开关（集中在这里）
    args.use_cluster_emb = True
    args.use_cluster_hist = False
    args.cluster_id_dropout = 0.0

    return args


def main():
    args = parse_args()
    seed_all(args.seed)

    args.exp_dir = build_exp_dir(args)
    args.log_dir = os.path.join(args.exp_dir, "logs")
    args.ckpt_dir = os.path.join(args.exp_dir, f"ckpts/fold-{args.k}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.overwrite and args.phase == "train" and args.k == 0 and os.path.exists(args.exp_dir):
        # 轻量 overwrite（避免你原来 rm -rvf 的危险写法）
        import shutil
        shutil.rmtree(args.exp_dir)
        os.makedirs(args.exp_dir, exist_ok=True)

    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(os.path.join(args.log_dir, f"{args.phase}-stdout-fold{args.k}.txt"))

    # ✅ fold 内全局聚类（生成 coord pkl）
    run_fold_clustering(args)

    # train/test
    runner = DMINMIL(args)
    metrics = runner.train() if args.phase == "train" else runner.test()

    # save metrics
    csv_path = os.path.join(args.log_dir, "test_metrics.csv")
    write_header = (args.k == 0) and (not os.path.exists(csv_path))
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["fold", "best_test_loss", "best_test_auc", "best_test_acc",
                        "best_test_precision", "best_test_recall", "best_test_f1"])
        w.writerow([args.k] + [round(float(m), 4) for m in metrics])

    logging.info(f"[Done] metrics saved to {csv_path}")


if __name__ == "__main__":
    main()