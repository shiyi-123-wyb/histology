# Parallel_processing_handler.py
# DYNAMIC LOAD BALANCING VERSION - FIXED FOR PICKLE ERRORS
# This version uses multiprocessing-safe queues and creates locks inside workers
import os
import shutil
import traceback
from pathlib import Path

# PyTorch imports
import torch
import torch.multiprocessing as mp

# Custom module imports   
from Dataset import WSIDataset   
from DeepCluster_framework import DeepCluster   

# GPU ID mapper
def create_gpu_mapping(specified_gpu_ids):
    """Create mapping from PyTorch GPU indices to actual hardware GPU IDs with validation"""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return {}
    
    total_gpus = torch.cuda.device_count()
    print(f"\nTotal GPUs available: {total_gpus}")
    
    if specified_gpu_ids is None:
        return {i: i for i in range(total_gpus)}
    
    valid_gpu_ids = []
    invalid_gpu_ids = []
    
    for gpu_id in specified_gpu_ids:
        if 0 <= gpu_id < total_gpus:
            valid_gpu_ids.append(gpu_id)
        else:
            invalid_gpu_ids.append(gpu_id)
    
    if invalid_gpu_ids:
        print(f"WARNING: Invalid GPU IDs removed: {invalid_gpu_ids}")
        print(f"Available GPU range: 0-{total_gpus-1}")
    
    if not valid_gpu_ids:
        print("No valid GPU IDs specified, falling back to GPU 0")
        valid_gpu_ids = [0] if total_gpus > 0 else []
    
    gpu_mapping = {}
    for pytorch_idx, actual_gpu_id in enumerate(valid_gpu_ids):
        gpu_mapping[pytorch_idx] = actual_gpu_id
    
    return gpu_mapping 

def get_actual_gpu_id(pytorch_gpu_id):
    """Convert PyTorch GPU ID to actual hardware GPU ID"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        visible_gpus = [int(x.strip()) for x in cuda_visible.split(',') if x.strip().isdigit()]
        if pytorch_gpu_id < len(visible_gpus):
            return visible_gpus[pytorch_gpu_id]
    return pytorch_gpu_id

def has_no_subfolders(path) -> bool:
    p = Path(path)
    return p.is_dir() and not any(child.is_dir() for child in p.iterdir())

# GPU worker assignment
def gpu_worker_with_queue(gpu_id, wsi_queue, results_dict, config, log_file, gpu_mapping, stop_event):     
    try:
        csv_lock = mp.Lock()
        
        # Validate GPU
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        elif gpu_id >= torch.cuda.device_count():
            print(f"Invalid GPU ID {gpu_id}, using CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
        
        actual_gpu_id = gpu_mapping.get(gpu_id, gpu_id) if gpu_mapping else get_actual_gpu_id(gpu_id) 
        
        processed_count = 0
        
        while not stop_event.is_set():
            try:
                # Try to get next WSI from queue (with timeout to check stop_event)
                wsi_path = wsi_queue.get(timeout=1)
                
                # Get WSI name for logging
                wsi_name = os.path.basename(wsi_path).split(',')[0].strip()
                
                # print(f"\n[GPU {actual_gpu_id}] Starting WSI: {wsi_name} (Queue: {wsi_queue.qsize()} remaining)")
                
                # Process the WSI
                try:
                    result = DeepCluster(wsi_path, device, config, log_file, actual_gpu_id, csv_lock)
                    
                    if result:
                        processed_count += 1
                        results_dict[wsi_name] = {'success': True, 'gpu': actual_gpu_id}
                        # print(f"[GPU {actual_gpu_id}] ✓ Completed WSI: {wsi_name} (Total: {processed_count})")
                    else:
                        results_dict[wsi_name] = {'success': False, 'gpu': actual_gpu_id, 'error': 'Processing failed'}
                        # print(f"[GPU {actual_gpu_id}] ✗ Failed WSI: {wsi_name}")
                
                except Exception as e:
                    print(f"[GPU {actual_gpu_id}] Error processing {wsi_name}: {e}")
                    traceback.print_exc()
                    results_dict[wsi_name] = {'success': False, 'gpu': actual_gpu_id, 'error': str(e)}
                
                finally:
                    wsi_queue.task_done()
                    
            except Exception:
                # Queue.get() timed out or queue is empty
                if wsi_queue.empty():
                    # Queue is truly empty, exit
                    break
        
        # print(f"\n[GPU {actual_gpu_id}] Worker finished. Processed {processed_count} WSIs.")
        
    except Exception as e:
        print(f"Critical error in GPU worker {gpu_id}: {e}")
        traceback.print_exc()
    
    finally:
        # Clear GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

# Processing input folders in parallel using GPU
def process_all_input_folders_parallel(config, device, log_file, specified_gpu_ids=None): 
    
    try:
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to serial processing")
            return process_all_input_folders_serial(config, device, log_file)
        
        # Prepare dataset
        if has_no_subfolders(config.input_path) or config.process_all:
            input_folders_list = [config.input_path]          
        else:
            dataset = WSIDataset(config.input_path, config.selected_input_folders, log_file)
            input_folders_list = dataset.slide_files 
    
        if not input_folders_list:
            print(f"\nNo folders found to process in {config.input_path}\n")
            return
    
        # Create GPU mapping
        gpu_mapping = create_gpu_mapping(specified_gpu_ids)
        
        if not gpu_mapping:
            print("No valid GPUs available, falling back to serial processing")
            return process_all_input_folders_serial(config, device, log_file)
        
        # Setup GPUs
        if specified_gpu_ids:
            total_gpus = torch.cuda.device_count()
            valid_gpu_ids = [gpu_id for gpu_id in specified_gpu_ids if 0 <= gpu_id < total_gpus]
            
            if not valid_gpu_ids:
                print("No valid GPU IDs, using all available GPUs")
                device_ids = list(range(total_gpus))
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, valid_gpu_ids))
                device_ids = list(range(len(valid_gpu_ids)))
        else:
            device_ids = list(range(torch.cuda.device_count()))

        num_gpus = min(len(device_ids), len(input_folders_list))
        
        if num_gpus == 0:
            print("No GPUs available, falling back to serial processing")
            return process_all_input_folders_serial(config, device, log_file)
        
        print(f"GPU's in action : {num_gpus} ") 
        print(f"Total WSIs to process: {len(input_folders_list)} \n") 
        
        # Create shared Manager and its objects
        manager = mp.Manager()
        
        # CRITICAL FIX: Use manager.Queue() instead of queue.Queue()
        # queue.Queue() is for threading, not multiprocessing
        wsi_queue = manager.Queue()  # FIXED: was queue.Queue()
        
        for wsi_path in input_folders_list:
            wsi_queue.put(wsi_path)
        
        # print(f"\nAdded {wsi_queue.qsize()} WSIs to processing queue\n")
               
        # Shared results dictionary and stop event
        results_dict = manager.dict()
        stop_event = mp.Event()
        
        # Launch GPU workers
        print('-' * 100)
        print("DeepCluster++ started...!!!")  
        print('-' * 100)
        
        processes = []
        
        for pytorch_gpu_id in range(num_gpus):
            if pytorch_gpu_id >= len(device_ids):
                continue
            
            process = mp.Process(
                target=gpu_worker_with_queue,
                args=(pytorch_gpu_id, wsi_queue, results_dict, config, log_file, gpu_mapping, stop_event))
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        try:
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user! Stopping workers...")
            stop_event.set()
            for process in processes:
                process.terminate()
                process.join(timeout=5)
        
        print("\n" + "=" * 100)
        print("DeepCluster++ completed!!!")  
        print("=" * 100)
        
        # Print summary
        successful = sum(1 for r in results_dict.values() if r.get('success', False))
        failed = len(results_dict) - successful
        
        print(f"\nProcessing Summary:")
        print(f"  ✓ Successful: {successful}")
        print(f"  ✗ Failed: {failed}")
        
        # Show per-GPU distribution
        from collections import Counter
        gpu_distribution = Counter(r['gpu'] for r in results_dict.values() if 'gpu' in r)
        print(f"\nWSIs processed per GPU:")
        for gpu_id, count in sorted(gpu_distribution.items()):
            print(f"  GPU {gpu_id}: {count} WSIs")
        
        # Clean up
        temp_features_base = os.path.join(config.output_path, 'temp_features')
        if os.path.exists(temp_features_base):
            try: 
                shutil.rmtree(temp_features_base)
                print(f"\nCleaned up all temporary feature directories\n")
            except Exception as e:
                print(f"Warning: Could not clean up temporary feature base directory: {e}") 
        print("=" * 100)
        print()
    except Exception as e:
        print(f"Error during processing: {str(e)}")        
        traceback.print_exc()

# Processing input folders serially using CPU
def process_all_input_folders_serial(config, device, log_file):    
    """Original serial processing - kept for backward compatibility"""
    try:          
        
        if has_no_subfolders(config.input_path) or config.process_all:
            input_folders_list = [config.input_path]          
        else:
            dataset = WSIDataset(config.input_path, config.selected_input_folders, log_file)
            input_folders_list = dataset.slide_files 
    
        if not input_folders_list:
            print(f"\nNo folders found to process in {config.input_path}\n")
            return
            
        successful = 0
        failed = 0 
 
        print("DeepCluster++ started...!!!")  
        print('-' * 100)
            
        for input_folder_path in input_folders_list:
            success = DeepCluster(input_folder_path, device, config, log_file, actual_gpu_id=None, csv_lock=None)
            if success:
                successful += 1
            else:
                failed += 1  
                
        print(' ')
        print('-' * 100) 
        print("DeepCluster++ completed...!!!")  
        print('-' * 100)
        
        temp_features_base = os.path.join(config.output_path, 'temp_features')        
        if os.path.exists(temp_features_base):
            try: 
                shutil.rmtree(temp_features_base)
                print(f"\nCleaned up all temporary feature directories\n")
            except Exception as e:
                print(f"Warning: Could not clean up temporary feature base directory: {e}")
            
        print(f"Successfully processed: {successful} input_folders (WSIs)\n")
        if failed > 0:
            print(f"Failed to process: {failed} input_folders (WSIs)\n")
        print('-' * 100)
        print(' ')
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")        
        traceback.print_exc()