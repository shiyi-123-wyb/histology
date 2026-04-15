# Standard library imports 
import os, csv
from datetime import datetime

# Creating a log file
def create_log_file(config, log_file):
    # Write header to the CSV file (only once, before the parallel process starts)
    file_exists = os.path.isfile(log_file)
    os.makedirs(os.path.dirname(log_file), exist_ok=True) 

    if config.sub_folders:
        sub_folders = [name.strip() for name in config.sub_folders.split(',')]
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # If the file does not exist, write the header
            if not file_exists:
                header = ["WSI_Name"]
            
                # Add dynamic part: each sub_folder + "Number_of_samples"
                for sf in sub_folders:
                    header.extend([sf])
            
                # Add the fixed columns
                header.extend(["Total_images", "Num_clusters", "Num_samples_selected", "Feature_extraction_time(s)", "Clustering_time(s)", "Sampling_time(s)", "Overall_time(s)", "Date_time_processed", "Additional_note"])
                writer.writerow(header)
    else:
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # If the file does not exist, write the header
            if not file_exists:      
                writer.writerow(["WSI_Name", "Total_images", "Num_clusters", "Num_samples_selected", "Feature_extraction_time(s)", "Clustering_time(s)", "Sampling_time(s)", "Overall_time(s)", "Date_time_processed", "Additional_note"]) 

# Here lock is multiprocessing object
def write_metrics_to_log_parallel(log_file, folder_name, metrics_dict, csv_lock):
    """Write metrics for a processed folder to the CSV log file""" 
    try:
        with csv_lock:
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                
                # Create row data based on whether sub_folders are specified
                row_data = [folder_name]
                
                # Add sub_folder counts if sub_folders are specified
                if 'sub_folder_counts' in metrics_dict:
                    for count in metrics_dict['sub_folder_counts']:
                        row_data.append(count)
                
                # Add standard metrics
                row_data.extend([ metrics_dict.get('Total_samples', 0), metrics_dict.get('Num_clusters', 0), metrics_dict.get('Num_samples_selected', 0), metrics_dict.get('Feature_extraction_time', 0.0),
                    metrics_dict.get('Clustering_time', 0.0), metrics_dict.get('Sampling_time', 0.0), metrics_dict.get('Overall_time', 0.0), metrics_dict.get('Date_time_processed', ''), metrics_dict.get('Additional_note', '')
                ])
                
                writer.writerow(row_data)
                
    except Exception as e:
        print(f"Warning: Could not write metrics to log file for {folder_name}: {e}")
        # Don't raise the exception to avoid stopping the entire process

# Here lock is boolean value
def write_metrics_to_log_serial(log_file, folder_name, metrics_dict):
    """Write metrics for a processed folder to the CSV log file""" 
    try: 
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Create row data based on whether sub_folders are specified
            row_data = [folder_name]
                
            # Add sub_folder counts if sub_folders are specified
            if 'sub_folder_counts' in metrics_dict:
                for count in metrics_dict['sub_folder_counts']:
                    row_data.append(count) 
                
            # Add standard metrics
            row_data.extend([metrics_dict.get('Total_samples', 0), metrics_dict.get('Num_clusters', 0), metrics_dict.get('Num_samples_selected', 0), metrics_dict.get('Feature_extraction_time', 0.0),
                    metrics_dict.get('Clustering_time', 0.0), metrics_dict.get('Sampling_time', 0.0), metrics_dict.get('Overall_time', 0.0), metrics_dict.get('Date_time_processed', ''), metrics_dict.get('Additional_note', '')
            ])
                
            writer.writerow(row_data)
                
    except Exception as e:
        print(f"Warning: Could not write metrics to log file for {folder_name}: {e}")
        # Don't raise the exception to avoid stopping the entire process