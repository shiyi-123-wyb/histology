# Standard library imports
import os, argparse, warnings
import numpy as np
import shutil  # 用于文件复制

# PyTorch imports
import torch

# Custom module imports
from Config import Config
from Parallel_processing_handler import process_all_input_folders_parallel
from Parallel_processing_handler import process_all_input_folders_serial
from Logging import create_log_file

# Filter warnings
warnings.filterwarnings("ignore")
# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disable oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Converting command line arguments from string to boolean
def str2bool(s: str) -> bool:
    if s.lower() in {'yes', 'true', 't', '1'}:
        return True
    if s.lower() in {'no', 'false', 'f', '0'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


def main():
    # Hardcoded parameters (replacing argparse)
    args = type('Args', (), {})()  # Simple object for args
    args.input_path = '/media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/patches_x20'
    args.output_path = '/media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/outputss'
    args.feature_ext = 'ae_crc'
    args.gpu_ids = '0'
    args.device = 'all_gpus'
    args.use_gpu_clustering = True  # 若cuML OK；否则False
    args.batch_size = 128  # 安全值
    args.dim_reduce = 256
    args.distance_groups = 5
    args.sample_percentage = 0.20
    args.model = '/home/joyivan/Downloads/AE_CRC.pth'
    args.seed = 42
    args.store_features = True
    args.store_clusters = True
    args.store_plots = True
    args.store_samples = True
    args.store_samples_group_wise = True
    args.test_mode = False   # 打开测试模式
    args.batch_size_wsi = 10  # 只跑前 10 个 WSI


    # Data loading verification: Print input path details
    print("=== Data Loading Verification ===")
    if os.path.exists(args.input_path):
        print(f"Input path exists: {args.input_path}")
        input_folders = [f for f in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, f))]
        print(f"Number of WSI folders detected: {len(input_folders)}")
        print(f"Sample WSI folders (first 5): {input_folders[:5]}")

        # Check a sample folder for patches
        if input_folders:
            sample_folder = os.path.join(args.input_path, input_folders[0])
            sample_patches = [f for f in os.listdir(sample_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Sample WSI '{input_folders[0]}' has {len(sample_patches)} patches")
            if sample_patches:
                print(f"Sample patches (first 3): {sample_patches[:3]}")
            else:
                print("Warning: No image patches found in sample folder!")
        else:
            print("Error: No WSI folders found in input path!")
            return
    else:
        print(f"Error: Input path does not exist: {args.input_path}")
        return

    # Check model path
    if os.path.exists(args.model):
        print(f"Model file exists: {args.model}")
    else:
        print(f"Warning: Model file not found: {args.model}")
        return

    # 自动复制模型到预期路径
    encoders_dir = os.path.join(os.path.dirname(__file__), 'Encoders')
    expected_model_path = os.path.join(encoders_dir, 'AE_CRC.pth')
    os.makedirs(encoders_dir, exist_ok=True)
    if not os.path.exists(expected_model_path):
        shutil.copy2(args.model, expected_model_path)
        print(f"Model copied to expected path: {expected_model_path}")
    else:
        print(f"Model already at expected path: {expected_model_path}")

    # Check output path
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output path ready: {args.output_path}")
    print("=== End Verification ===")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 动态设置selected_input_folders (per-WSI模式)
    if args.test_mode:
        selected_folders = input_folders[:args.batch_size_wsi]  # 测试前50
        print(f"Test mode: Processing first {len(selected_folders)} WSIs")
    else:
        selected_folders = input_folders  # 全量
        print(f"Full mode: Processing all {len(selected_folders)} WSIs")
    args.selected_input_folders = ','.join(selected_folders)  # 逗号分隔传入handler

    # Create config object (process_all=False for per-WSI)
    config = Config(
        input_path=args.input_path,
        selected_input_folders=args.selected_input_folders,  # 关键: 指定所有WSI
        sub_folders=None,
        output_path=args.output_path,
        process_all=False,  # 关键: per-folder parallel
        batch_size=args.batch_size,
        device=args.device,
        feature_ext=args.feature_ext,
        dim_reduce=args.dim_reduce,
        num_distance_groups=args.distance_groups,
        sample_percentage=args.sample_percentage,
        store_features=args.store_features,
        store_clusters=args.store_clusters,
        store_plots=args.store_plots,
        store_samples=args.store_samples,
        store_samples_group_wise=args.store_samples_group_wise,
        use_gpu_clustering=args.use_gpu_clustering
    )

    # File to write metrics
    log_file = args.output_path + '/Summary.csv'
    create_log_file(config, log_file)

    # 原生Parallel处理 (GPU batch per WSI)
    print("=== Native Parallel Acceleration ===")
    print(f"Using parallel handler for {len(selected_folders)} WSIs on GPU(s): {args.gpu_ids}")

    device = None
    if args.gpu_ids is not None:
        print("-----------------------------------------------------")
        print("Parallel Processing Mode with GPUs")
        print("-----------------------------------------------------")
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

        # Validate GPU IDs
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            valid_gpu_ids = [gpu_id for gpu_id in gpu_ids if 0 <= gpu_id < available_gpus]
            invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id not in valid_gpu_ids]

            if invalid_gpu_ids:
                print(f"Warning: Invalid GPU IDs {invalid_gpu_ids} removed. Available GPUs: 0-{available_gpus - 1}")

            if valid_gpu_ids:
                print(f"GPU IDs: {','.join(map(str, valid_gpu_ids))}")
                for gpu_id in valid_gpu_ids:
                    print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
                gpu_ids = valid_gpu_ids
            else:
                print("No valid GPU IDs provided. Falling back to CPU.")
                gpu_ids = []
        else:
            print("No CUDA GPUs available.")
            gpu_ids = []
        print("-----------------------------------------------------")

        if gpu_ids:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            process_all_input_folders_parallel(config, device, log_file, gpu_ids)
        else:
            device = torch.device('cpu')
            print("-----------------------------------------------------")
            print("Single-threaded Processing Mode (CPU fallback)")
            print("-----------------------------------------------------")
            print(f"Using device: {device}")
            print("-----------------------------------------------------")
            process_all_input_folders_serial(config, device, log_file)

    # All GPUs (if no gpu_ids)
    elif args.device == "all_gpus":
        print("-----------------------------------------------------")
        print("Parallel Processing Mode with GPUs")
        print("-----------------------------------------------------")

        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))
            print(f"GPU IDs: {','.join(map(str, gpu_ids))}")
            for gpu_id in gpu_ids:
                print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print("No CUDA GPUs available.")
            gpu_ids = []
        print("-----------------------------------------------------")

        if gpu_ids:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            process_all_input_folders_parallel(config, device, log_file, gpu_ids)
        else:
            device = torch.device('cpu')
            print("-----------------------------------------------------")
            print("Single-threaded Processing Mode (CPU fallback)")
            print("-----------------------------------------------------")
            print(f"Using device: {device}")
            print("-----------------------------------------------------")
            process_all_input_folders_serial(config, device, log_file)

    # Only CPU
    else:
        device = torch.device('cpu')
        print("-----------------------------------------------------")
        print("Single-threaded Processing Mode")
        print("-----------------------------------------------------")
        print(f"Using device: {device}")
        print("-----------------------------------------------------")

        process_all_input_folders_serial(config, device, log_file)

    print("\n=== Processing Complete ===")
    print(f"Check {args.output_path}/Summary.csv for per-WSI stats (clusters, samples).")
    print("To run full: set args.test_mode = False")


if __name__ == "__main__":
    main()