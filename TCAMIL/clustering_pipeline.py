import os
import logging
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix, diags


def read_split_cases(args):
    df = pd.read_csv(os.path.join(args.split_dir, f'splits_{args.k}.csv'))
    result = {}
    for split in ('train', 'val', 'test'):
        result[split] = {
            str(int(v)) for v in df[split].dropna()
        } if split in df.columns else set()
    return result['train'], result['val'], result['test']


def find_wsi_feature_and_cluster_files(root_dir):
    result = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "cluster_assignments.csv" not in filenames:
            continue

        wsi_id = os.path.basename(dirpath)
        cluster_csv = os.path.join(dirpath, "cluster_assignments.csv")

        feature_csv = None
        for fname in filenames:
            if fname.endswith(".csv") and fname != "cluster_assignments.csv":
                path = os.path.join(dirpath, fname)
                try:
                    head = pd.read_csv(path, nrows=5)
                except Exception:
                    continue
                if "File_Path" in head.columns and "Cluster" not in head.columns and head.shape[1] >= 10:
                    feature_csv = path
                    break

        if feature_csv is None:
            logging.warning(f"[Clustering] No feature csv found for WSI {wsi_id}, skipping.")
            continue

        result.append({"wsi": wsi_id, "feature_csv": feature_csv, "cluster_csv": cluster_csv})

    logging.info(f"[Clustering] Found {len(result)} WSIs with cluster_assignments.csv")
    return result


def parse_xy_from_filepath(file_path):
    name = os.path.splitext(os.path.basename(file_path))[0]

    m = re.search(r'x(\d+)_y(\d+)', name)
    if m:
        return int(m.group(1)), int(m.group(2))

    mx, my = re.search(r'x(\d+)', name), re.search(r'y(\d+)', name)
    if mx and my:
        return int(mx.group(1)), int(my.group(1))

    nums = [p for p in re.split(r'[_\-]', name) if p.isdigit()]
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])

    logging.warning(f"[Clustering] Could not parse (x, y) from: {file_path}")
    return None


def _compute_replication_factors(sizes, mode="sqrt", alpha=0.5, base_rep=1,
                                  max_rep=10, min_rep=1, max_total_rep=200000):
    sizes = np.maximum(np.asarray(sizes, dtype=np.float64), 1.0)
    mode = (mode or "sqrt").lower()

    if mode == "none":
        return np.ones(len(sizes), dtype=np.int64)

    if mode == "sqrt":
        w = np.sqrt(sizes)
    elif mode == "log":
        w = np.log1p(sizes)
    elif mode in ("linear", "pow"):
        w = np.power(sizes, alpha if alpha is not None else (1.0 if mode == "linear" else 0.5))
    else:
        logging.warning(f"[Clustering] Unknown center_weight_mode={mode}, falling back to sqrt.")
        w = np.sqrt(sizes)

    med = np.median(w)
    if not np.isfinite(med) or med <= 0:
        med = 1.0

    rep = np.clip(np.rint((w / (med + 1e-12)) * base_rep).astype(np.int64), min_rep, max_rep)

    total = int(rep.sum())
    if total > max_total_rep:
        scale = max_total_rep / float(total)
        rep = np.clip(np.maximum(min_rep, np.floor(rep * scale)).astype(np.int64), min_rep, max_rep)
        logging.info(f"[Clustering] Replication capped: {total} -> {int(rep.sum())} (scale={scale:.4f})")

    return rep


def _majority_vote_labels(rep_labels, rep_to_orig, n_orig):
    rep_labels  = np.asarray(rep_labels,  dtype=np.int64)
    rep_to_orig = np.asarray(rep_to_orig, dtype=np.int64)
    out = np.zeros(n_orig, dtype=np.int64)
    for i in range(n_orig):
        idx = np.where(rep_to_orig == i)[0]
        out[i] = Counter(rep_labels[idx].tolist()).most_common(1)[0][0] if idx.size else 0
    return out


def _is_new_coord_map_format(obj):
    # New format: dict[wsi] -> dict[(x,y)] -> cluster
    # Old format: dict[(x,y)] -> cluster
    if not isinstance(obj, dict) or len(obj) == 0:
        return isinstance(obj, dict)
    return isinstance(next(iter(obj.values())), dict)


class LSCModel:
    """Landmark-based Spectral Clustering."""

    def __init__(self, k, p=128, r=5, mode="random", random_state=42, max_iter=100, n_init=10):
        self.k = k
        self.p = p
        self.r = r
        self.mode = mode
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init

        self.marks = self.sigma = self.feaSum = self.inv_feaSum = None
        self.S = self.Vt = self.kmeans = None
        self.is_fitted = False

    def _select_landmarks(self, X):
        p = min(self.p, X.shape[0])
        if self.mode == "random":
            idx = np.random.RandomState(self.random_state).choice(X.shape[0], size=p, replace=False)
            return X[idx]
        if self.mode == "kmeans":
            km = KMeans(n_clusters=p, random_state=self.random_state, n_init=5, max_iter=20, verbose=0)
            km.fit(X)
            return km.cluster_centers_
        raise ValueError(f"Unsupported LSC mode: {self.mode}")

    def _build_Z(self, X, marks, sigma=None, r=5):
        D = pairwise_distances(X, marks, metric="euclidean")

        if not sigma or sigma <= 0:
            sigma = np.mean(D) or 1.0

        n_smp, p = D.shape
        r = min(r, p)

        idx_r = np.argpartition(D, kth=r - 1, axis=1)[:, :r]
        dump = D[np.arange(n_smp)[:, None], idx_r]
        weights = np.exp(-dump / (2.0 * sigma ** 2))
        Gsdx = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)

        Z = coo_matrix(
            (Gsdx.ravel(), (np.repeat(np.arange(n_smp), r), idx_r.ravel())),
            shape=(n_smp, marks.shape[0]),
        ).tocsr()
        return Z, sigma

    def _compute_embedding_from_Z(self, Z_norm):
        if self.S is None or self.Vt is None:
            raise RuntimeError("S and Vt not set — call fit() first.")
        U_approx = (Z_norm.dot(self.Vt.T) if hasattr(Z_norm, "dot") else Z_norm @ self.Vt.T)
        U_approx /= (self.S + 1e-12)
        Y = U_approx[:, 1:]
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return Y / norms

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        steps = ["Selecting landmarks", "Building sparse Z", "Column-normalising Z",
                 "SVD on Z_norm", "Embedding + KMeans"]

        with tqdm(total=5, desc="[LSC] Global clustering", unit="step", ncols=120,
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}, {rate_fmt}]") as pbar:

            self.marks = self._select_landmarks(X); pbar.update(1)

            Z, self.sigma = self._build_Z(X, self.marks, r=self.r); pbar.update(1)

            feaSum = np.sqrt(np.array(Z.sum(axis=0)).ravel())
            feaSum[feaSum < 1e-12] = 1e-12
            self.feaSum = feaSum
            self.inv_feaSum = 1.0 / feaSum
            Z_norm = Z.dot(diags(self.inv_feaSum)); pbar.update(1)

            Z_dense = Z_norm.toarray() if hasattr(Z_norm, "toarray") else np.asarray(Z_norm)
            U, S, Vt = np.linalg.svd(Z_dense, full_matrices=False)
            n_comp = self.k + 1
            self.S, self.Vt = S[:n_comp], Vt[:n_comp]; pbar.update(1)

            Y = self._compute_embedding_from_Z(Z_norm)
            self.kmeans = KMeans(n_clusters=self.k, random_state=self.random_state,
                                 n_init=self.n_init, max_iter=self.max_iter, verbose=0)
            self.kmeans.fit(Y)
            self.is_fitted = True; pbar.update(1)

        return self.kmeans.labels_

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("LSCModel must be fit before calling transform().")
        Z_new, _ = self._build_Z(np.asarray(X, dtype=np.float64), self.marks, sigma=self.sigma, r=self.r)
        return self._compute_embedding_from_Z(Z_new.dot(diags(self.inv_feaSum)))

    def predict(self, X):
        return self.kmeans.predict(self.transform(X))


def run_fold_clustering(args):
    root_dir = args.cluster_root
    res_dir  = os.path.join(root_dir, f"0.87/global_clustering_fold_{args.k}")
    os.makedirs(res_dir, exist_ok=True)

    coord_pkl_train = os.path.join(res_dir, "coord_to_global_cluster_trainval.pkl")
    coord_pkl_test  = os.path.join(res_dir, "coord_to_global_cluster_test.pkl")
    model_pkl       = os.path.join(res_dir, "global_cluster_model.pkl")

    # Return early if valid cache exists
    if all(os.path.exists(p) for p in (model_pkl, coord_pkl_train, coord_pkl_test)):
        try:
            with open(coord_pkl_train, "rb") as f:
                cached = pickle.load(f)
            if _is_new_coord_map_format(cached):
                logging.info(f"[Clustering] Fold-{args.k}: using cached global clustering results.")
                args.coord_pkl_train_path = coord_pkl_train
                args.coord_pkl_test_path  = coord_pkl_test
                args.coord_pkl_path       = coord_pkl_train
                return
            logging.warning(f"[Clustering] Fold-{args.k}: old coord map format detected, recomputing.")
        except Exception as e:
            logging.warning(f"[Clustering] Cache check failed ({e}), recomputing.")

    logging.info(f"[Clustering] Fold-{args.k}: running global clustering under {res_dir}")

    train_cases, val_cases, test_cases = read_split_cases(args)
    train_val_cases = train_cases | val_cases
    logging.info(f"[Clustering] fold-{args.k}: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")

    wsi_files = find_wsi_feature_and_cluster_files(root_dir)

    centers_train, meta_train = [], []
    centers_test,  meta_test  = [], []

    for rec in tqdm(wsi_files, desc="[Clustering] Collecting local cluster centers"):
        wsi_id = str(rec["wsi"])
        if wsi_id not in train_val_cases and wsi_id not in test_cases:
            continue

        try:
            df_feat = pd.read_csv(rec["feature_csv"])
            df_clu  = pd.read_csv(rec["cluster_csv"])
        except Exception as e:
            logging.warning(f"[Clustering] Failed to read WSI {wsi_id}: {e}")
            continue

        required = {"File_Path"}
        if not required <= set(df_feat.columns) or not (required | {"Cluster"}) <= set(df_clu.columns):
            logging.warning(f"[Clustering] Column mismatch for WSI {wsi_id}")
            continue

        df = pd.merge(df_clu, df_feat, on="File_Path", how="inner")
        if df.empty:
            logging.warning(f"[Clustering] Empty merge for WSI {wsi_id}")
            continue

        feat_cols = [c for c in df.columns if c not in ("File_Path", "Cluster")]
        bucket    = (centers_train, meta_train) if wsi_id in train_val_cases else (centers_test, meta_test)

        for local_c, g in df.groupby("Cluster"):
            feats = g[feat_cols].values.astype(np.float32)
            if len(feats) == 0:
                continue
            bucket[0].append(feats.mean(axis=0))
            bucket[1].append({"wsi": wsi_id, "local_cluster": int(local_c),
                               "size": len(feats), "center_idx": int(g.index[0])})

    if not centers_train:
        raise RuntimeError("[Clustering] No train+val local centers collected.")

    X_train = np.vstack(centers_train)
    logging.info(f"[Clustering] Train+val centers: {X_train.shape}")

    global_pca_dim = getattr(args, "global_pca_dim", 32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if X_scaled.shape[1] > global_pca_dim:
        logging.info(f"[Clustering] PCA: {X_scaled.shape[1]} -> {global_pca_dim}")
        pca = PCA(n_components=global_pca_dim, random_state=args.seed)
        X_pca = pca.fit_transform(X_scaled)
    else:
        pca, X_pca = None, X_scaled

    cw = {
        "mode":      getattr(args, "center_weight_mode",      "sqrt"),
        "alpha":     getattr(args, "center_weight_alpha",     0.5),
        "base_rep":  getattr(args, "center_weight_base_rep",  1),
        "max_rep":   getattr(args, "center_weight_max_rep",   10),
        "max_total": getattr(args, "center_weight_max_total", 200000),
    }

    sizes = np.array([m["size"] for m in meta_train], dtype=np.float64)
    rep   = _compute_replication_factors(sizes, mode=cw["mode"], alpha=cw["alpha"],
                                         base_rep=cw["base_rep"], max_rep=cw["max_rep"],
                                         min_rep=1, max_total_rep=cw["max_total"])

    if cw["mode"].lower() != "none":
        logging.info(f"[Clustering] Size weighting: mode={cw['mode']}, total_rep={int(rep.sum())}")

    rep_to_orig = np.repeat(np.arange(len(X_pca), dtype=np.int64), rep)
    X_pca_rep   = X_pca[rep_to_orig]

    lsc = LSCModel(k=args.num_clusters,
                   p=getattr(args, "p_landmarks", 128),
                   r=getattr(args, "r_neighbors", 5),
                   mode=getattr(args, "lsc_mode", "random"),
                   random_state=args.seed, max_iter=100, n_init=10)

    logging.info(f"[Clustering] LSC: K={lsc.k}, p={lsc.p}, r={lsc.r}, mode={lsc.mode}")
    train_labels = _majority_vote_labels(lsc.fit(X_pca_rep), rep_to_orig, len(meta_train))

    for i, lab in enumerate(train_labels):
        meta_train[i]["global_cluster"] = int(lab)

    with open(model_pkl, "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca, "lsc": lsc, "weighting": cw}, f)
    logging.info(f"[Clustering] Model saved to {model_pkl}")

    if centers_test:
        X_test   = np.vstack(centers_test).astype(np.float64)
        X_test_t = scaler.transform(X_test)
        X_test_t = pca.transform(X_test_t) if pca is not None else X_test_t
        for i, lab in enumerate(lsc.predict(X_test_t)):
            meta_test[i]["global_cluster"] = int(lab)
        logging.info(f"[Clustering] Assigned global clusters to {len(centers_test)} test centers.")

    # Save local->global mapping
    df_local = pd.DataFrame(meta_train + meta_test)[
        ["wsi", "local_cluster", "size", "center_idx", "global_cluster"]
    ]
    local_csv = os.path.join(res_dir, "local_cluster_to_global.csv")
    df_local.to_csv(local_csv, index=False)
    logging.info(f"[Clustering] local_cluster_to_global saved: {df_local.shape}")

    local2global = {(str(r.wsi), int(r.local_cluster)): int(r.global_cluster)
                    for r in df_local.itertuples()}

    # Build patch-level mapping
    patch_records = []
    for rec in tqdm(wsi_files, desc="[Clustering] Building patch->global mapping"):
        wsi_id = str(rec["wsi"])
        try:
            df_clu = pd.read_csv(rec["cluster_csv"])
        except Exception as e:
            logging.warning(f"[Clustering] Failed to read cluster csv for WSI {wsi_id}: {e}")
            continue
        if "File_Path" not in df_clu.columns or "Cluster" not in df_clu.columns:
            continue
        for row in df_clu.itertuples(index=False):
            patch_records.append({
                "wsi": wsi_id, "File_Path": row.File_Path,
                "local_cluster": int(row.Cluster),
                "global_cluster": local2global.get((wsi_id, int(row.Cluster)), 0),
            })

    df_patch = pd.DataFrame(patch_records)
    patch_csv = os.path.join(res_dir, "patch_to_global_cluster.csv")
    df_patch.to_csv(patch_csv, index=False)
    logging.info(f"[Clustering] patch_to_global_cluster saved: {df_patch.shape}")

    # Build coord pkl (per-WSI buckets to avoid key collisions)
    coord_map_trainval, coord_map_test = {}, {}
    for row in df_patch.itertuples(index=False):
        wsi, fp, gc = str(row.wsi), row.File_Path, int(row.global_cluster)
        xy = parse_xy_from_filepath(fp)
        if xy is None:
            continue
        bucket = coord_map_trainval if wsi in train_val_cases else coord_map_test if wsi in test_cases else None
        if bucket is not None:
            bucket.setdefault(wsi, {})[xy] = gc

    for path, mapping in ((coord_pkl_train, coord_map_trainval), (coord_pkl_test, coord_map_test)):
        with open(path, "wb") as f:
            pickle.dump(mapping, f)
        n_coords = sum(len(v) for v in mapping.values())
        logging.info(f"[Clustering] Saved {path}: {len(mapping)} WSIs, {n_coords} coords")

    args.coord_pkl_train_path = coord_pkl_train
    args.coord_pkl_test_path  = coord_pkl_test
    args.coord_pkl_path       = coord_pkl_train
