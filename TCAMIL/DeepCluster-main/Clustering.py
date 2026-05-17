import os
import time
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd
from scipy.spatial import ConvexHull

GPU_AVAILABLE = False
try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    GPU_AVAILABLE = True
except ImportError:
    pass

from Config import Config


def calc_num_clusters(n_samples):
    return math.ceil(0.1 * math.sqrt(n_samples))


def _pbar(desc, total=1, unit="step"):
    return tqdm(
        total=total, desc=desc, unit=unit, leave=True, ncols=120,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]",
    )


def _anchor_spectral_clustering(X, n_clusters, n_anchors=None, n_neighbors=5, random_state=42):
    n_samples, _ = X.shape

    if n_samples <= n_clusters:
        return np.zeros(n_samples, dtype=np.int32)

    if n_anchors is None:
        n_anchors = min(max(2 * n_clusters, 32), max(n_clusters, 256), n_samples)

    anchor_km = MiniBatchKMeans(
        n_clusters=n_anchors, random_state=random_state,
        batch_size=min(2048, n_samples), max_iter=50, n_init=1,
    )
    anchor_km.fit(X)
    anchors = anchor_km.cluster_centers_

    n_neighbors = min(n_neighbors, n_anchors)
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(anchors)
    distances, indices = nn.kneighbors(X)

    sigma = np.mean(distances[:, -1]) + 1e-8
    rows = np.repeat(np.arange(n_samples), n_neighbors)
    weights = np.exp(-(distances.ravel() ** 2) / (2.0 * sigma ** 2)).astype(np.float32)

    Z = np.zeros((n_samples, n_anchors), dtype=np.float32)
    Z[rows, indices.ravel()] = weights

    row_sums = Z.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    B = Z / np.sqrt(row_sums)

    n_comp = min(n_clusters, B.shape[0], B.shape[1])
    if n_comp < 1:
        fallback = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=random_state,
            batch_size=min(2048, n_samples), max_iter=100, n_init=3,
        )
        return fallback.fit_predict(X)

    U, _, _ = randomized_svd(B, n_components=n_comp, random_state=random_state)
    Y = U[:, :n_clusters]
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    Y = np.nan_to_num(Y)

    final_km = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state,
        batch_size=min(2048, n_samples), max_iter=100, n_init=5,
    )
    return final_km.fit_predict(Y).astype(np.int32)


def perform_kmeans(X, n_clusters, folder_name, use_gpu, device, actual_gpu_id=None):
    tag = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    desc = f"|{folder_name}| - |{tag}| - |Spectral Clustering ({n_clusters} clusters)|"

    with _pbar(desc, total=100, unit="%") as pbar:
        pbar.n = 10
        pbar.refresh()
        clusters = _anchor_spectral_clustering(X, n_clusters)
        pbar.n = 100
        pbar.refresh()

    return clusters


def perform_pca(features_scaled, n_components, folder_name, use_gpu, device, actual_gpu_id=None):
    tag = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    desc = f"|{folder_name}| - |{tag}| - |PCA (512 -> {n_components} dims)|"

    if use_gpu and GPU_AVAILABLE and device.type == "cuda":
        with _pbar(desc) as pbar:
            try:
                features_gpu = cp.asarray(features_scaled, dtype=cp.float32)
                reduced = cp.asnumpy(cuPCA(n_components=n_components, random_state=42).fit_transform(features_gpu))
                pbar.update(1)
                return reduced
            except Exception as e:
                print(f"  GPU PCA failed: {e}, falling back to CPU")

    with _pbar(desc) as pbar:
        reduced = PCA(n_components=n_components, random_state=42).fit_transform(features_scaled)
        pbar.update(1)
    return reduced


def perform_pca_visualization(features_reduced, folder_name, use_gpu, device, actual_gpu_id=None):
    tag = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    n_in = features_reduced.shape[1]
    desc = f"|{folder_name}| - |{tag}| - |PCA Visualization ({n_in} -> 2 dims)|"

    if use_gpu and GPU_AVAILABLE and device.type == "cuda":
        with _pbar(desc) as pbar:
            try:
                features_gpu = cp.asarray(features_reduced, dtype=cp.float32)
                result = cp.asnumpy(cuPCA(n_components=2, random_state=42).fit_transform(features_gpu))
                pbar.update(1)
                return result
            except Exception as e:
                print(f"  GPU PCA visualization failed: {e}, falling back to CPU")

    with _pbar(desc) as pbar:
        result = PCA(n_components=2, random_state=42).fit_transform(features_reduced)
        pbar.update(1)
    return result


def copy_image_parallel(args):
    file_path, output_path = args
    try:
        if os.path.exists(file_path):
            Image.open(file_path).save(output_path)
            return True
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
    return False


def process_clusters(folder_name, feature_file, n_samples, config: Config, temp_feature_dir, device, actual_gpu_id=None):
    tag = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)

    try:
        data = pd.read_csv(feature_file)
        features = data.drop("File_Path", axis=1).values.astype(np.float32)
        file_paths = data["File_Path"].values

        features_scaled = StandardScaler().fit_transform(features)
        features_scaled = np.nan_to_num(features_scaled, nan=0.0001, posinf=0.0001, neginf=0.0001)

        n_components = config.dim_reduce
        if n_samples <= n_components:
            n_clusters = 1
            clusters = np.zeros(n_samples, dtype=int)
            features_reduced = features_scaled
        else:
            features_reduced = perform_pca(
                features_scaled, n_components, folder_name,
                config.use_gpu_clustering, device, actual_gpu_id,
            )
            n_clusters = calc_num_clusters(n_samples)
            clusters = perform_kmeans(
                features_reduced, n_clusters, folder_name,
                config.use_gpu_clustering, device, actual_gpu_id,
            )

        if config.store_plots and n_samples > 2 and n_clusters > 1:
            plot_dir = os.path.join(config.plot_dir, folder_name)
            os.makedirs(plot_dir, exist_ok=True)

            features_reduced = np.nan_to_num(features_reduced, nan=0.0001, posinf=0.0001, neginf=0.0001)
            features_2d = perform_pca_visualization(
                features_reduced, folder_name, config.use_gpu_clustering, device, actual_gpu_id,
            )
            palette = sns.color_palette("husl", n_colors=n_clusters)
            visualize_clusters(features_2d, clusters, file_paths, os.path.join(plot_dir, "pca_visualization"), palette)

        assignments = pd.DataFrame({"File_Path": file_paths, "Cluster": clusters})

        save_dir = os.path.dirname(feature_file) if config.store_features else temp_feature_dir
        assignments.to_csv(os.path.join(save_dir, "cluster_assignments.csv"), index=False)

        if config.store_clusters:
            cluster_base = os.path.join(config.cluster_dir, folder_name)
            os.makedirs(cluster_base, exist_ok=True)
            desc = f"|{folder_name}| - |{tag}| - |Saving cluster data|"
            max_workers = min(8, mp.cpu_count())

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for cluster in tqdm(range(n_clusters), desc=desc, unit="cluster", leave=True, ncols=120,
                                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                               "[{elapsed}<{remaining}, {rate_fmt}]"):
                    cluster_dir = os.path.join(cluster_base, f"Cluster_{cluster}")
                    os.makedirs(cluster_dir, exist_ok=True)
                    cluster_files = file_paths[clusters == cluster]
                    args = [
                        (fp, os.path.join(cluster_dir, os.path.basename(fp)))
                        for fp in cluster_files if os.path.exists(fp)
                    ]
                    list(executor.map(copy_image_parallel, args))

        return assignments, n_clusters

    except Exception as e:
        print(f"Error in process_clusters: {e}")
        raise


def visualize_clusters(features_2d, clusters, file_paths, output_prefix, palette):
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    plt.switch_backend("Agg")

    def draw_hulls(ax, points, color):
        if len(points) < 5:
            return
        x_range, y_range = np.ptp(points[:, 0]), np.ptp(points[:, 1])
        if x_range < 1e-6 or y_range < 1e-6:
            return
        try:
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], c=color)
        except Exception:
            pass

    # Legend plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        pts = features_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[palette[cluster]], label=f"Cluster {cluster}", alpha=0.6)
        draw_hulls(ax, pts, palette[cluster])
    ax.set_title("PCA Visualization with Spectral Clusters")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_with_legend.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    # Numbered plot
    fig, ax = plt.subplots(figsize=(16, 11))
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        pts = features_2d[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[palette[cluster]], alpha=0.6)
        if len(pts) > 0:
            cx, cy = pts.mean(axis=0)
            ax.text(cx, cy, str(cluster), ha="center", va="center", fontweight="bold")
        draw_hulls(ax, pts, palette[cluster])
    ax.set_title("PCA Visualization with Cluster Numbers")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_with_numbers.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
