# Standard library imports 
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Custom module imports
from Config import Config 

def copy_single_image(args):
    """Helper function to copy a single image"""
    file_path, output_path = args
    try:
        img = Image.open(file_path)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return False

# Sample images from clusters with parallel processing
def sample_clustered_images(folder_name: str, n_clusters: int, config: Config, temp_feature_dir: str, device=None, actual_gpu_id=None):   

    # Get GPU information for display
    device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    
    try:
        # Determine where to read files from
        if config.store_features:
            subfolder_feature_dir = os.path.join(config.feature_dir, folder_name)
            assignment_file = os.path.join(subfolder_feature_dir, "cluster_assignments.csv")
            feature_file = os.path.join(subfolder_feature_dir, "features.csv")
        else:
            assignment_file = os.path.join(temp_feature_dir, "cluster_assignments.csv")
            feature_file = os.path.join(temp_feature_dir, "features.csv")
        
        if not os.path.exists(assignment_file) or not os.path.exists(feature_file):
            print(f"Missing required files for sample selection in {folder_name}")
            return
        
        # Load data
        assignments = pd.read_csv(assignment_file)
        features_df = pd.read_csv(feature_file)
        
        # Prepare output directory
        samples_base = os.path.join(config.output_path, 'samples', folder_name)
        os.makedirs(samples_base, exist_ok=True)
        
        cluster_ids = assignments['Cluster'].unique()
        
        # Process clusters in parallel
        max_workers = min(4, mp.cpu_count() // 2)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for cluster_id in cluster_ids:
                future = executor.submit(process_single_cluster, cluster_id, assignments, features_df, samples_base, folder_name, config)
                futures.append((cluster_id, future))
                
            progress_desc = f"|{folder_name}| - |{device_display}| - |Sampling images|"
            
            # Wait for completion with progress bar
            for cluster_id, future in tqdm(futures, 
                                          desc = progress_desc, unit = "cluster", leave = True, ncols = 120,
                                          bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing cluster {cluster_id}: {e}")
    
    except Exception as e:
        print(f"Error in sample_clustered_images: {str(e)}")
        raise

def process_single_cluster(cluster_id, assignments, features_df, samples_base, folder_name, config):
    """Process a single cluster - optimized for parallel execution"""
    try:
        # Get files and features for this cluster
        cluster_mask = assignments['Cluster'] == cluster_id
        cluster_files = assignments.loc[cluster_mask, 'File_Path'].values
        
        if len(cluster_files) == 0:
            return
        
        # Create output directory
        cluster_sample_dir = os.path.join(samples_base, f'Cluster_{cluster_id}')
        os.makedirs(cluster_sample_dir, exist_ok=True)
        
        # Vectorized feature matching
        cluster_files_set = set(cluster_files)
        cluster_features_df = features_df[features_df['File_Path'].isin(cluster_files_set)]
        cluster_features = cluster_features_df.drop('File_Path', axis=1).values
        cluster_file_paths = cluster_features_df['File_Path'].values
        
        if len(cluster_features) == 0:
            return
        
        # Calculate centroid and distances (vectorized)
        centroid = np.mean(cluster_features, axis=0)
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        
        # Normalize distances
        if len(distances) > 1:
            max_dist = distances.max()
            min_dist = distances.min()
            
            if max_dist == min_dist:
                normalized_distances = np.zeros_like(distances)
            else:
                normalized_distances = (distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_distances = np.zeros(1)
        
        # Equal-frequency binning
        sorted_indices = np.argsort(normalized_distances)
        total_samples = len(sorted_indices)
        
        samples_per_group = total_samples // config.num_distance_groups
        remainder = total_samples % config.num_distance_groups
        
        distance_groups = []
        current_idx = 0
        
        for group_idx in range(config.num_distance_groups):
            extra = 1 if group_idx < remainder else 0
            group_size = samples_per_group + extra
            
            if group_idx == config.num_distance_groups - 1:
                group_size = total_samples - current_idx
            
            group_indices = sorted_indices[current_idx:current_idx + group_size]
            group_files = cluster_file_paths[group_indices]
            distance_groups.append(group_files)
            current_idx += group_size
        
        # Prepare all copy tasks
        all_copy_tasks = []
        
        for group_idx, group_files in enumerate(distance_groups):
            if len(group_files) == 0:
                continue
            
            num_samples = max(1, int(len(group_files) * config.sample_percentage))
            
            selected_indices = np.random.choice(len(group_files), 
                                               size=min(num_samples, len(group_files)), 
                                               replace=False)
            selected_files = group_files[selected_indices]
            
            if config.store_samples_group_wise:
                group_dir = os.path.join(cluster_sample_dir, f'Group_{group_idx}')
                os.makedirs(group_dir, exist_ok=True)
                output_dir = group_dir
            else:
                output_dir = cluster_sample_dir
            
            for file_path in selected_files:
                output_path = os.path.join(output_dir, os.path.basename(file_path))
                all_copy_tasks.append((file_path, output_path))
        
        # Execute all copies in parallel
        if all_copy_tasks:
            with ThreadPoolExecutor(max_workers=4) as copy_executor:
                list(copy_executor.map(copy_single_image, all_copy_tasks))
    
    except Exception as e:
        print(f"Error processing cluster {cluster_id}: {e}")
        raise