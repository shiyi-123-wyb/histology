import os, argparse, warnings, shutil
import numpy as np
import torch

from Config import Config
from Parallel_processing_handler import process_all_input_folders_parallel, process_all_input_folders_serial
from Logging import create_log_file

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def str2bool(s: str) -> bool:
    if s.lower() in {'yes', 'true', 't', '1'}:
        return True
    if s.lower() in {'no', 'false', 'f', '0'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


def resolve_gpu_ids(gpu_ids_str):
    """Parse gpu_ids string, validate against available hardware, return list of valid ints."""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available.")
        return []

    available = torch.cuda.device_count()
    requested = [int(x.strip()) for x in gpu_ids_str.split(",")]
    valid = [g for g in requested if 0 <= g < available]
    invalid = [g for g in requested if g not in valid]

    if invalid:
        print(f"Warning: GPU IDs {invalid} out of range (available: 0-{available - 1}), ignoring.")

    for g in valid:
        print(f"  GPU {g}: {torch.cuda.get_device_name(g)}")

    return valid


def run(config, gpu_ids, log_file):
    if gpu_ids:
        device = torch.device('cuda')
        process_all_input_folders_parallel(config, device, log_file, gpu_ids)
    else:
        device = torch.device('cpu')
        process_all_input_folders_serial(config, device, log_file)


def main():
    args = type('Args', (), {
        'input_path':               '/media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/patches_x20',
        'output_path':              '/media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/outputss',
        'feature_ext':              'ae_crc',
        'gpu_ids':                  '0',
        'device':                   'all_gpus',
        'use_gpu_clustering':       True,
        'batch_size':               128,
        'dim_reduce':               256,
        'distance_groups':          5,
        'sample_percentage':        0.20,
        'seed':                     42,
        'store_features':           True,
        'store_clusters':           False,
        'store_plots':              False,
        'store_samples':            False,
        'store_samples_group_wise': False,
        'test_mode':                False,
        'batch_size_wsi':           10,
    })()

    # Validate input
    if not os.path.exists(args.input_path):
        print(f"Error: input path not found: {args.input_path}")
        return

    input_folders = [
        f for f in os.listdir(args.input_path)
        if os.path.isdir(os.path.join(args.input_path, f))
    ]
    if not input_folders:
        print("Error: no WSI folders found.")
        return

    sample_dir = os.path.join(args.input_path, input_folders[0])
    n_patches = sum(1 for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg')))
    print(f"Found {len(input_folders)} WSIs. '{input_folders[0]}' has {n_patches} patches.")

    # Copy model if needed
    encoders_dir = os.path.join(os.path.dirname(__file__), 'Encoders')
    expected_model = os.path.join(encoders_dir, 'AE_CRC.pth')
    os.makedirs(encoders_dir, exist_ok=True)
    if not os.path.exists(expected_model):
        shutil.copy2(args.model, expected_model)
        print(f"Model copied to {expected_model}")

    os.makedirs(args.output_path, exist_ok=True)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # WSI selection
    selected = input_folders[:args.batch_size_wsi] if args.test_mode else input_folders
    print(f"{'Test' if args.test_mode else 'Full'} mode: processing {len(selected)} WSIs.")

    config = Config(
        input_path=args.input_path,
        selected_input_folders=','.join(selected),
        sub_folders=None,
        output_path=args.output_path,
        process_all=False,
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
        use_gpu_clustering=args.use_gpu_clustering,
    )

    log_file = os.path.join(args.output_path, 'Summary.csv')
    create_log_file(config, log_file)

    if args.gpu_ids is not None:
        gpu_ids = resolve_gpu_ids(args.gpu_ids)
    elif args.device == 'all_gpus' and torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
        for g in gpu_ids:
            print(f"  GPU {g}: {torch.cuda.get_device_name(g)}")
    else:
        gpu_ids = []

    run(config, gpu_ids, log_file)

    print(f"\nDone. Results at {args.output_path}/Summary.csv")


if __name__ == "__main__":
    main()
