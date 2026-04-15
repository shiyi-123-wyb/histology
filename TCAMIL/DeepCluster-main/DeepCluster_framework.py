# Standard library imports
import os
import time
import shutil
from datetime import datetime

# Third-party imports
import torch

# Custom function imports  
from Dataset import count_images_input_folder
from Feature_extraction import extract_features
from Clustering import process_clusters
from Sampling import sample_clustered_images
from Logging import write_metrics_to_log_parallel, write_metrics_to_log_serial 

# Global variables
image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}

# Counting the number of images
def count_images(input_path):    
    counter = 0
    if os.path.exists(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    counter += 1
    return counter

# DeepCluster algorithm
def DeepCluster(slide_path, device, config, log_file, actual_gpu_id=None, csv_lock=None):
    
    folder_name = os.path.basename(slide_path)
    temp_feature_dir = None 
    
    # Initialize timing
    start_time = time.time()
    feature_extraction_time = 0.0
    clustering_time = 0.0
    sampling_time = 0.0

    try:  
        # Count images
        n_samples = count_images_input_folder(slide_path, config)

        if n_samples == 0:
            print(f"No images found in folder: {slide_path}. Skipping.")
            return False
  
        # Display info
        device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
        print(f"\n|{folder_name}| - |{device_display}| - |{n_samples} samples| - |Processing started!!!|")

        # Feature extraction using AutoEncoder
        feature_start = time.time()
        result = extract_features(slide_path, folder_name, n_samples, config, device, actual_gpu_id)
        feature_extraction_time = time.time() - feature_start
        
        if len(result) == 3:
            feature_file, actual_samples, temp_feature_dir = result
        else:
            feature_file, actual_samples = result
            temp_feature_dir = None
      
        if feature_file is not None:
            # K-means clustering  
            clustering_start = time.time()
            assignments, n_clusters = process_clusters(folder_name, feature_file, actual_samples, config, temp_feature_dir or "", device, actual_gpu_id)
            clustering_time = time.time() - clustering_start
            
            # Sampling using equal frequency binning
            sampling_start = time.time()
            num_samples_selected = 0
            if config.store_samples: 
                sample_clustered_images(folder_name, n_clusters, config, temp_feature_dir or "", device, actual_gpu_id)
                
                # Count sampled images
                samples_folder = os.path.join(config.output_path, "samples", folder_name)
                num_samples_selected = count_images(samples_folder) 
            sampling_time = time.time() - sampling_start

            # Calculate metrics
            overall_time = time.time() - start_time
            date_time_processed = datetime.now().astimezone() 

            # Prepare metrics dictionary
            metrics_dict = {
                'Total_samples': actual_samples, 'Num_clusters': n_clusters, 'Num_samples_selected': num_samples_selected,
                'Feature_extraction_time': round(feature_extraction_time, 2), 'Clustering_time': round(clustering_time, 2),
                'Sampling_time': round(sampling_time, 2), 'Overall_time': round(overall_time, 2),
                'Date_time_processed': str(date_time_processed.strftime("%Y-%m-%d %H:%M:%S %Z")),
                'Additional_note': f'GPU:{actual_gpu_id}' if actual_gpu_id is not None else 'CPU'
            }
            
            # Add sub_folder counts if applicable
            if config.sub_folders:
                sub_folder_names = [name.strip() for name in config.sub_folders.split(',')]
                sub_folder_counts = []
                for sub_folder in sub_folder_names:
                    sub_path = os.path.join(slide_path, sub_folder)
                    if os.path.exists(sub_path):
                        sub_count = count_images(sub_path)
                        sub_folder_counts.append(sub_count)
                    else:
                        sub_folder_counts.append(0)
                metrics_dict['sub_folder_counts'] = sub_folder_counts 

            # Write metrics to log
            if str(device) == 'cpu': 
                write_metrics_to_log_serial(log_file, folder_name, metrics_dict)

            if csv_lock: 
                write_metrics_to_log_parallel(log_file, folder_name, metrics_dict, csv_lock)
                
            print(f"|{folder_name}| - |{device_display}| - |{n_samples} samples| - |Processing is completed!!!|")  
            
            # Clean up temporary files
            if temp_feature_dir:  
                cleanup_temporary_files(temp_feature_dir, config)            
            return True
        else:
            print(f"Feature extraction failed for subfolder: {folder_name}. Skipping.")
            return False
            
    except Exception as e:
        print(f"Error processing WSI {folder_name}: {str(e)}")         
        # Clean up on error
        if temp_feature_dir:
            cleanup_temporary_files(temp_feature_dir, config)        
        return False
        
    finally:
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if temp_feature_dir: 
            cleanup_temporary_files(temp_feature_dir, config)      

def cleanup_temporary_files(temp_feature_dir, config):
    """Clean up temporary feature files if store_features is False"""
    if os.path.exists(temp_feature_dir):
        try: 
            shutil.rmtree(temp_feature_dir)
        except Exception as e:
            pass