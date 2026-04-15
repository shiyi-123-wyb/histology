# Standard library imports
import os
import csv
from tqdm import tqdm
from pathlib import Path 
from PIL import Image

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 

# Custom imports  
from Dataset import TileDataset
from Config import Config 

# Encoders
from Encoders.AECRCEncoder import AECRCEncoder
from Encoders.ResNet50Encoder import ResNet50Encoder
from Encoders.ResNet18Encoder import ResNet18Encoder
from Encoders.DenseNet121Encoder import DenseNet121Encoder
from Encoders.EfficientNetB0Encoder import EfficientNetB0Encoder
from Encoders.EfficientNetB7Encoder import EfficientNetB7Encoder 
from Encoders.ViTEncoder import ViTEncoder
from Encoders.CustomCNNEncoder import CustomCNNEncoder
from Encoders.UNIEncoder import UNIEncoder
from Encoders.CONCHEncoder import CONCHEncoder
from Encoders.ProvGigaPathEncoder import ProvGigaPathEncoder
from Encoders.CTransPathEncoder import CTransPathEncoder
from Encoders.ResNet50_1024Encoder import ResNet50_1024Encoder
from Encoders.UNI2Encoder import UNI2Encoder
from Encoders.dinov2_extractor import DinoV2Extractor

#To avoid ValueError: Decompressed data too large for PngImagePlugin.MAX_TEXT_CHUNK
os.environ['PIL_MAX_TEXT_CHUNK'] = '10485760'  # 100MB
from PIL import PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 100MB

# Add an encoder file in Encoders folder, import above, and include in the following dictionary
Available_Encoders = { 
    'ae_crc' : AECRCEncoder,
    'resnet50': ResNet50Encoder,
    'resnet50_1024': ResNet50_1024Encoder,
    'resnet18': ResNet18Encoder,
    'densenet121': DenseNet121Encoder,
    'efficientnet_b0': EfficientNetB0Encoder,
    'efficientnet_b7': EfficientNetB7Encoder,
    'vit_b16': ViTEncoder,
    'custom_cnn': CustomCNNEncoder,
    'uni': UNIEncoder,
    'conch': CONCHEncoder,
    'prov_gigapath': ProvGigaPathEncoder,
    'ctranspath': CTransPathEncoder,
    'uni_v2' : UNI2Encoder,
    'dinov2' : DinoV2Extractor
}

# Encoder factory for easy instantiation
class TileDataset(Dataset):
    """Dataset class for loading tiles from folders with robust error handling"""
    
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.image_files = []
        self.transform = transform or transforms.ToTensor()
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']        
        self._collect_image_files()
    
    def _collect_image_files(self):
        """Collect all valid image files from the folder structure"""
        if not self.folder_path.exists():
            print(f"Warning: Folder path does not exist: {self.folder_path}")
            return
            
        for root, dirs, files in os.walk(self.folder_path):
            root_path = Path(root)
            for file_name in files:
                if any(file_name.lower().endswith(ext) for ext in self.image_extensions):
                    img_path = root_path / file_name
                    class_label = root_path.relative_to(self.folder_path)
                    if str(class_label) == '.':
                        class_label = self.folder_path.name
                    self.image_files.append((str(img_path), str(class_label))) 
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path, class_label = self.image_files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, IOError, Exception) as e:
            #print(f"[WARNING] Skipping corrupted image: {img_path}")
            return None
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            
        return img, class_label, img_path 
        
# Loading encoder
def get_encoder_class(encoder_name):
    """Factory function to create encoder instances"""
    if encoder_name.lower() not in Available_Encoders:
        available = ', '.join(Available_Encoders.keys())
        raise ValueError(f"Unknown encoder '{encoder_name}'. Available encoders: {available}")
    
    encoder_class = Available_Encoders[encoder_name.lower()]
    return encoder_class 


def custom_collate_fn(batch):
    """Custom collate function to filter out None values from corrupted images"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None
    return torch.stack([item[0] for item in batch]), \
           [item[1] for item in batch], \
           [item[2] for item in batch]

    
# Feature extraction using encoder from AutoEncoder
def extract_features(input_folder_path: str, folder_name: str, n_samples: int, config: Config, device: torch.device, actual_gpu_id=None):
    
    # Get GPU information for display
    device_display = f"GPU:{actual_gpu_id}" if actual_gpu_id is not None else str(device)
    encoder_name = config.feature_extractor

    # Create encoder with proper device handling
    encoder_class = get_encoder_class(encoder_name) 
    encoder = encoder_class(config, actual_gpu_id)
    feature_dim = encoder.feature_dim
    
    dataset = TileDataset(folder_path = input_folder_path, transform = encoder.transform)
 
    if len(dataset) == 0:
        print(f"No images found in {input_folder_path}. Skipping.")
        return None, 0

    # Optimized DataLoader settings
    num_workers = 4 if encoder.device.type == 'cuda' else 2
    
    data_loader = DataLoader(dataset = dataset, batch_size = config.batch_size, shuffle = False, num_workers = num_workers,
        pin_memory = True if encoder.device.type == 'cuda' else False, persistent_workers = True if num_workers > 0 else False,
        prefetch_factor = 4 if num_workers > 0 else None, collate_fn=custom_collate_fn)
    
    # Create temporary feature directory
    temp_feature_dir = os.path.join(config.output_path, 'temp_features', folder_name)
    os.makedirs(temp_feature_dir, exist_ok=True)
    
    # Determine where to save features
    if config.store_features:
        folder_feature_dir = os.path.join(config.feature_dir, folder_name)
        os.makedirs(folder_feature_dir, exist_ok=True)
        feature_file = os.path.join(folder_feature_dir, "features.csv")
    else:
        feature_file = os.path.join(temp_feature_dir, "features.csv")
    
    progress_desc = f"|{folder_name}| - |{device_display}| - |Features Extraction|"
    
    with open(feature_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = [f"Feature_{i}" for i in range(1, feature_dim + 1)] + ["File_Path"]
        writer.writerow(header)
        
        # Model is already on encoder.device, don't move it again
        encoder.model.eval()
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc = progress_desc, unit = "batch", leave = True, ncols = 120,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for batch_idx, (images, labels, paths) in enumerate(progress_bar):
                try:
                    # Skip if batch is None (from collate function)
                    if images is None:
                        continue
                    
                    valid_mask = images.sum(dim=[1, 2, 3]) != 0
                    valid_images = images[valid_mask]
                    valid_paths = [p for p, m in zip(paths, valid_mask) if m]
                    
                    # Skip empty batches
                    if len(valid_images) == 0:
                        continue
                    
                    batch = valid_images.to(encoder.device, non_blocking=True)
                    
                    # Extract features
                    features = encoder.extract_features(batch)
                    
                    # Features are already numpy array, no need to call .cpu()
                    latent_cpu = features
                    
                    # Batch write to CSV
                    rows = [list(feature_vector) + [path] for feature_vector, path in zip(latent_cpu, valid_paths)]
                    writer.writerows(rows)
                    
                    # Update progress less frequently
                    if batch_idx % 10 == 0:
                        processed_samples = (batch_idx + 1) * config.batch_size
                        if encoder.device.type == 'cuda':
                            memory_info = f'{torch.cuda.memory_allocated(encoder.device) // 1024**2}MB'
                        else:
                            memory_info = 'N/A'
                        progress_bar.set_postfix({
                            'Samples': f'{min(processed_samples, len(dataset))}/{len(dataset)}',
                            'Memory': memory_info
                        })
                    
                    # Clear cache less frequently
                    if batch_idx % 100 == 0 and encoder.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing batch {batch_idx} in {folder_name}: {str(e)}")
                    continue
       
    return feature_file, len(dataset), temp_feature_dir