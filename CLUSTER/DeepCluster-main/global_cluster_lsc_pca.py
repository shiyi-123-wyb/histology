import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils.extmath import randomized_svd

from scipy.sparse import coo_matrix, diags


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global clustering: PCA→32 + Landmark-based Spectral Clustering (LSC)"
    )

    #  per-WSI 特征 & cluster_assignments.csv 所在的根目录
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/outputss",
        help="Root directory that contains per-WSI feature folders with cluster_assignments.csv"
    )

    # 全局簇数
    parser.add_argument(
        "--num_global_clusters",
        type=int,
        default=24,
        help="Number of global clusters (K_global)"
    )

    # PCA 降到多少维再做 LSC
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=32,
        help="PCA dimension before LSC (e.g., 32)"
    )

    # LSC 参数：landmark 数量 p、每个样本连接的最近 landmark 数 r
    parser.add_argument(
        "--p_landmarks",
        type=int,
        default=128,
        help="Number of landmarks p in LSC (default 128)"
    )
    parser.add_argument(
        "--r_neighbors",
        type=int,
        default=5,
        help="Number of nearest landmarks r for each sample in LSC (default 5)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="random",
        choices=["random", "kmeans"],
        help="Landmark selection mode: random or kmeans (default random)"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


# ---------- LSC 主函数：Python 版 LSC，实现跟 LSC.m 类似 ----------

def lsc_clustering(
    X,
    k,
    p=128,
    r=5,
    mode="random",
    random_state=42,
    max_iter=100,
    n_init=10
):
    """
    Python 实现的 Landmark-based Spectral Clustering (LSC).
    输入:
        X: [n_samples, n_features] 的数据矩阵 (这里已经是 PCA 后的低维表示)
        k: 聚成 k 个簇 (全局簇数)
        p: landmarks 数量
        r: 每个样本连接的最近 landmarks 数
        mode: 'random' or 'kmeans' (landmark 选取方式)
    输出:
        labels: [n_samples]，每个局部簇中心的 global_cluster_id
    """
    np.random.seed(random_state)
    n_smp, n_fea = X.shape

    # -------- 1. 选 landmarks ----------
    p = min(p, n_smp)
    if mode == "random":
        idx = np.random.choice(n_smp, size=p, replace=False)
        marks = X[idx]
    elif mode == "kmeans":
        # 用 KMeans 的中心作为 landmarks
        km = KMeans(
            n_clusters=p,
            random_state=random_state,
            n_init=5,
            max_iter=20,
            verbose=0
        )
        km.fit(X)
        marks = km.cluster_centers_
    else:
        raise ValueError("Unsupported mode for LSC: {}".format(mode))

    # -------- 2. 计算 X 到 landmarks 的距离矩阵 D (n_smp x p) ----------
    # 跟 MATLAB LSC.m 保持一致，用 Euclidean distance
    D = pairwise_distances(X, marks, metric="euclidean")

    # sigma: 均值距离
    sigma = np.mean(D)
    if sigma <= 0:
        sigma = 1.0

    # -------- 3. 构造 Z (n_smp x p) 的稀疏表示 ----------
    # 对每个样本，找 r 个最近的 landmarks
    if r > p:
        r = p

    # idx_r: [n_smp, r] 最近 landmarks 的索引
    idx_r = np.argpartition(D, kth=r-1, axis=1)[:, :r]
    dump = D[np.arange(n_smp)[:, None], idx_r]   # [n_smp, r]

    # 权重 = exp(-d / (2*sigma^2))  (注意: LSC.m 里是 -dump/(2*sigma^2)，没有平方)
    weights = np.exp(-dump / (2 * sigma * sigma))
    sumD = np.sum(weights, axis=1, keepdims=True) + 1e-12
    Gsdx = weights / sumD

    # 用 COO 构造稀疏矩阵 Z
    row_idx = np.repeat(np.arange(n_smp), r)
    col_idx = idx_r.reshape(-1)
    data_vals = Gsdx.reshape(-1)

    Z = coo_matrix((data_vals, (row_idx, col_idx)), shape=(n_smp, p)).tocsr()

    # -------- 4. 列归一化 (跟 LSC.m 一致) ----------
    feaSum = np.sqrt(np.array(Z.sum(axis=0)).ravel())
    feaSum[feaSum < 1e-12] = 1e-12
    inv_feaSum = 1.0 / feaSum
    Z = Z.dot(diags(inv_feaSum, offsets=0))

    # -------- 5. 对 Z 做 SVD，取前 k+1 个奇异向量 ----------
    # 使用 randomized_svd，对稀疏矩阵更稳
    U, S, Vt = randomized_svd(
        Z,
        n_components=k+1,
        n_iter=5,
        random_state=random_state
    )
    # 丢掉第一个奇异向量
    U = U[:, 1:]   # [n_smp, k]

    # 行归一化
    row_norm = np.linalg.norm(U, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    U_norm = U / row_norm

    # -------- 6. 在 U_norm 空间上再做一次 KMeans ----------
    km_final = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        verbose=0
    )
    labels = km_final.fit_predict(U_norm)
    return labels


# ---------- 扫描 root_dir，找到每个 WSI 的 features.csv + cluster_assignments.csv ----------

def find_wsi_feature_and_cluster_files(root_dir):
    """
    在 root_dir 下递归找所有包含 cluster_assignments.csv 的文件夹，
    并在同一文件夹中找对应的 features.csv（列中包含很多特征的那个 CSV）。

    返回 list，每个元素是:
        {
          'wsi': <wsi_id>,
          'feature_csv': <path>,
          'cluster_csv': <path>
        }
    """
    result = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "cluster_assignments.csv" not in filenames:
            continue

        cluster_csv = os.path.join(dirpath, "cluster_assignments.csv")

        # 从当前文件夹名推 WSI id（例如 .../features_ae_crc/449117）
        wsi_id = os.path.basename(dirpath)

        # 在同一个文件夹里找特征 csv（排除 cluster_assignments.csv）
        feature_csv_candidates = [
            f for f in filenames
            if f.endswith(".csv") and f != "cluster_assignments.csv"
        ]

        feature_csv = None
        for fname in feature_csv_candidates:
            path = os.path.join(dirpath, fname)
            try:
                df_head = pd.read_csv(path, nrows=5)
            except Exception:
                continue

            # 特征文件一般包含 'File_Path' 列，但不包含 'Cluster' 列，并且有很多数值特征列
            if "File_Path" in df_head.columns and "Cluster" not in df_head.columns:
                # 简单判断至少有 10 个维度的特征列
                if df_head.shape[1] >= 10:
                    feature_csv = path
                    break

        if feature_csv is None:
            print(f"[WARN] 找到 {cluster_csv}，但没找到对应的 features csv，跳过该 WSI: {wsi_id}")
            continue

        result.append({
            "wsi": wsi_id,
            "feature_csv": feature_csv,
            "cluster_csv": cluster_csv
        })

    print(f"总共找到 {len(result)} 个 WSI 的特征和 cluster_assignments.csv")
    return result


# ---------- 主流程 ----------

def main():
    args = parse_args()

    root_dir = args.root_dir
    num_global_clusters = args.num_global_clusters
    pca_dim = args.pca_dim
    p_landmarks = args.p_landmarks
    r_neighbors = args.r_neighbors
    mode = args.mode
    random_state = args.random_state

    np.random.seed(random_state)

    print("Root dir:", root_dir)
    print("num_global_clusters:", num_global_clusters)
    print("PCA dim:", pca_dim)
    print("LSC: p =", p_landmarks, ", r =", r_neighbors, ", mode =", mode)

    # -------- 1. 找到所有 WSI 的 features + cluster_assignments 文件 ----------
    wsi_files = find_wsi_feature_and_cluster_files(root_dir)
    if len(wsi_files) == 0:
        print("在 root_dir 下没有找到任何包含 cluster_assignments.csv 的文件夹，检查路径是否正确。")
        return

    # -------- 2. 计算所有局部簇中心 (local cluster centers) ----------
    all_centers = []
    meta_records = []   # 对应每个簇中心的 wsi, local_cluster, size, center_idx

    print("\n=== Step 1: 计算每个 WSI 的局部簇中心 ===")
    for rec in tqdm(wsi_files):
        wsi_id = rec["wsi"]
        feature_csv = rec["feature_csv"]
        cluster_csv = rec["cluster_csv"]

        try:
            df_feat = pd.read_csv(feature_csv)
            df_clu = pd.read_csv(cluster_csv)
        except Exception as e:
            print(f"[WARN] 读取失败, wsi={wsi_id}, error={e}")
            continue

        if "File_Path" not in df_feat.columns or "File_Path" not in df_clu.columns or "Cluster" not in df_clu.columns:
            print(f"[WARN] 列不匹配, wsi={wsi_id}, feat_cols={df_feat.columns}, clu_cols={df_clu.columns}")
            continue

        df = pd.merge(df_clu, df_feat, on="File_Path", how="inner")
        if df.shape[0] == 0:
            print(f"[WARN] wsi={wsi_id} merge 后没有样本，跳过")
            continue

        feature_cols = [c for c in df.columns if c not in ["File_Path", "Cluster"]]
        feat_matrix = df[feature_cols].values.astype(np.float32)

        # 按 local_cluster 分组，计算簇中心
        for local_c, g in df.groupby("Cluster"):
            local_c = int(local_c)
            g_feats = g[feature_cols].values.astype(np.float32)
            if g_feats.shape[0] == 0:
                continue

            center_vec = g_feats.mean(axis=0)
            cluster_size = g_feats.shape[0]

            # center_idx 这里简单设成簇内的第一个样本行号（没实际用到也没关系）
            center_idx = int(g.index[0])

            all_centers.append(center_vec)
            meta_records.append({
                "wsi": wsi_id,
                "local_cluster": local_c,
                "size": int(cluster_size),
                "center_idx": center_idx
            })

    if len(all_centers) == 0:
        print("没有任何簇中心被计算出来，检查 per-WSI 的 features/cluster_assignments 是否正常。")
        return

    X_centers = np.vstack(all_centers)   # [n_local_clusters, feat_dim]
    print(f"总局部簇数: {X_centers.shape[0]}, 特征维度: {X_centers.shape[1]}")

    # -------- 3. PCA 降维到 pca_dim，然后做 LSC ----------
    print("\n=== Step 2: 标准化 + PCA 降维到 {} 维 ===".format(pca_dim))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_centers)

    if X_scaled.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        X_pca = pca.fit_transform(X_scaled)
        print(f"PCA 后形状: {X_pca.shape}")
    else:
        X_pca = X_scaled
        print(f"原维度 <= pca_dim={pca_dim}，跳过 PCA，直接用标准化后的特征做 LSC。")

    print("\n=== Step 3: 在 PCA 后特征上做 LSC (Landmark-based Spectral Clustering) ===")
    labels = lsc_clustering(
        X_pca,
        k=num_global_clusters,
        p=p_landmarks,
        r=r_neighbors,
        mode=mode,
        random_state=random_state
    )
    assert labels.shape[0] == X_centers.shape[0]

    # 把 global_cluster 加到 meta_records
    for i, lab in enumerate(labels):
        meta_records[i]["global_cluster"] = int(lab)

    df_local = pd.DataFrame(meta_records)
    df_local = df_local[["wsi", "local_cluster", "size", "center_idx", "global_cluster"]]

    # -------- 4. 保存 local_cluster_to_global.csv ----------
    global_res_dir = os.path.join(root_dir, "global_clustering_results")
    os.makedirs(global_res_dir, exist_ok=True)

    local_csv_path = os.path.join(global_res_dir, "local_cluster_to_global.csv")
    df_local.to_csv(local_csv_path, index=False)
    print(f"\nSaved local cluster -> global cluster mapping to: {local_csv_path}")
    print(f"local_cluster_to_global.csv 形状: {df_local.shape}")

    # -------- 5. 构造 (wsi, patch) -> global_cluster，生成 patch_to_global_cluster.csv ----------
    print("\n=== Step 4: 生成 patch_to_global_cluster.csv ===")

    # 建立字典 (wsi, local_cluster) -> global_cluster
    local2global = {}
    for _, row in df_local.iterrows():
        key = (str(row["wsi"]), int(row["local_cluster"]))
        local2global[key] = int(row["global_cluster"])

    patch_records = []

    for rec in tqdm(wsi_files):
        wsi_id = rec["wsi"]
        cluster_csv = rec["cluster_csv"]

        try:
            df_clu = pd.read_csv(cluster_csv)
        except Exception as e:
            print(f"[WARN] 读取失败, wsi={wsi_id}, error={e}")
            continue

        if "File_Path" not in df_clu.columns or "Cluster" not in df_clu.columns:
            print(f"[WARN] 列不匹配 (patch-level), wsi={wsi_id}, clu_cols={df_clu.columns}")
            continue

        for _, row in df_clu.iterrows():
            fp = row["File_Path"]
            lc = int(row["Cluster"])
            key = (str(wsi_id), lc)
            if key not in local2global:
                # 理论上不应该发生；如果发生，就先设成 0 并报警
                # 但是不会影响文件格式
                print(f"[WARN] (wsi={wsi_id}, local_cluster={lc}) 在 local2global 里找不到，对应 global_cluster=0")
                gc = 0
            else:
                gc = local2global[key]

            patch_records.append({
                "wsi": str(wsi_id),
                "File_Path": fp,
                "local_cluster": lc,
                "global_cluster": gc
            })

    df_patch = pd.DataFrame(patch_records)
    patch_csv_path = os.path.join(global_res_dir, "patch_to_global_cluster.csv")
    df_patch.to_csv(patch_csv_path, index=False)
    print(f"Saved patch -> global cluster mapping to: {patch_csv_path}")
    print(f"patch_to_global_cluster.csv 形状: {df_patch.shape}")

    print("\n=== 全局聚类完成 (PCA→{} + LSC)，可以重新跑 coord_to_global_cluster.pkl 的脚本了 ===".format(pca_dim))


if __name__ == "__main__":
    main()
