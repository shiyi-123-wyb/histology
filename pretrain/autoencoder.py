import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
import openslide
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
import logging

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置日志
def setup_logging(log_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'autoencoder_training.log')),
            logging.StreamHandler()
        ]
    )

# 解码器
class Decoder(nn.Module):
    def __init__(self, in_channels=2048):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)

# 自编码器
class ResNet50Autoencoder(nn.Module):
    def __init__(self):
        super(ResNet50Autoencoder, self).__init__()
        resnet = resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 自定义数据集类：加载WSI的patch
class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, file_path, wsi, img_transforms, patch_size=256):
        self.file_path = file_path
        self.wsi = wsi
        self.img_transforms = img_transforms
        self.patch_size = patch_size
        with h5py.File(file_path, 'r') as f:
            self.coords = f['coords'][:]  # level=1 坐标
        self.length = len(self.coords)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        coord = self.coords[idx]
        x, y = coord[0], coord[1]
        patch = self.wsi.read_region((x, y), 1, (self.patch_size, self.patch_size))  # level=1
        patch = patch.convert('RGB')
        # 过滤背景 patch
        patch_np = np.array(patch)
        if patch_np.std() < 15:  # 严格阈值
            return self.__getitem__((idx + 1) % self.length)
        patch = self.img_transforms(patch)
        return {'img': patch, 'coord': torch.from_numpy(coord)}

# 自定义数据集类：加载折的训练集 slide ID
class Dataset_All_Bags(Dataset):
    def __init__(self, split_file, slide_ext='.svs'):
        try:
            self.df = pd.read_csv(split_file)
        except Exception as e:
            raise ValueError(f"Failed to read {split_file}: {e}")
        # 验证 train 列
        if 'train' not in self.df.columns:
            raise ValueError(f"'train' column not found in {split_file}. Columns: {self.df.columns.tolist()}")
        # 从 train 列提取非空 slide ID
        self.slide_ids = self.df['train'].dropna().astype(str).values
        logging.info(f"Fold {split_file}: Found {len(self.slide_ids)} training slides: {self.slide_ids[:5].tolist()}...")
        if len(self.slide_ids) == 0:
            logging.warning(f"No training slides found in {split_file}")
        self.slide_ext = slide_ext

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        return self.slide_ids[idx] + self.slide_ext

# 保存HDF5文件的工具函数
def save_hdf5(output_path, asset_dict, mode='a'):
    with h5py.File(output_path, mode) as f:
        for key, val in asset_dict.items():
            if mode == 'w' or key not in f:
                f.create_dataset(key, data=val, maxshape=(None, *val.shape[1:]))
            else:
                dataset = f[key]
                dataset.resize(dataset.shape[0] + val.shape[0], axis=0)
                dataset[-val.shape[0]:] = val

# 训练函数
def train_autoencoder(output_path, loader, model, criterion, optimizer, scheduler=None,
                      use_feature_loss=False, gradient_clip=None, verbose=1, slide_id=None, data_h5_dir=None):
    model.train()
    running_loss = 0.0
    file_exists = os.path.exists(output_path)
    history = []

    for count, data in enumerate(loader):
        batch = data['img'].to(device, non_blocking=True)
        coords = data['coord'].numpy().astype(np.int32)
        optimizer.zero_grad()
        outputs = model(batch)

        # 组合损失：像素 + 特征
        pixel_loss = criterion(outputs, batch)
        loss = pixel_loss
        if use_feature_loss:
            features = model.encoder(batch)
            recon_features = model.encoder(outputs)
            feature_loss = criterion(recon_features, features)
            loss += 0.5 * feature_loss

        loss.backward()
        if gradient_clip:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        running_loss += loss.item()
        history.append(loss.item())
        logging.info(f"Batch {count+1}/{len(loader)} Loss: {loss.item():.6f}")

        # 保存第一个 batch 的图像
        if count == 0:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
            original = batch * std + mean
            for i in range(original.size(0)):
                vutils.save_image(
                    original[i],
                    os.path.join(data_h5_dir, 'images', f'original_{slide_id}_batch1_patch{i+1}.png')
                )
                vutils.save_image(
                    outputs[i],
                    os.path.join(data_h5_dir, 'images', f'reconstructed_{slide_id}_batch1_patch{i+1}.png')
                )

        current_mode = 'w' if count == 0 and not file_exists else 'a'
        with torch.inference_mode():
            features = model.encoder(batch).cpu().numpy().astype(np.float32)
            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, mode=current_mode)

    avg_loss = running_loss / len(loader)
    loss_std = np.std(history)
    logging.info(f"Training statistics: Avg Loss={avg_loss:.6f}, Std={loss_std:.6f}")
    return avg_loss

# 主程序
def main(args):
    # 数据预处理
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 设置日志
    setup_logging(args.data_h5_dir)

    # 创建输出目录
    os.makedirs(os.path.join(args.data_h5_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(args.data_h5_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.data_h5_dir, 'pt'), exist_ok=True)

    # 验证 ID.csv
    try:
        id_df = pd.read_csv(args.csv_path)
        possible_id_cols = ['slide_id', 'case_id', 'ID']
        id_col = next((col for col in possible_id_cols if col in id_df.columns), id_df.columns[0])
        logging.info(f"ID.csv: Using ID column: {id_col}, Found {len(id_df)} slides")
        id_set = set(id_df[id_col].astype(str))
    except Exception as e:
        logging.error(f"Error reading {args.csv_path}: {e}")
        return

    # 十折训练
    for fold in range(10):
        logging.info(f"\nStarting Fold {fold}")
        split_file = os.path.join(args.splits_dir, f"splits_{fold}.csv")
        if not os.path.exists(split_file):
            logging.error(f"Split file {split_file} not found")
            continue

        # 加载训练集 slide
        try:
            bags_dataset = Dataset_All_Bags(split_file, args.slide_ext)
        except Exception as e:
            logging.error(f"Error loading {split_file}: {e}")
            continue
        if len(bags_dataset) == 0:
            logging.error(f"No training slides found in fold {fold}")
            continue

        # 验证 slide ID 是否在 ID.csv 中
        slide_ids = set(bags_dataset.slide_ids)
        common_ids = slide_ids.intersection(id_set)
        logging.info(f"Fold {fold}: {len(common_ids)}/{len(slide_ids)} slide IDs found in ID.csv")

        # 初始化模型和优化器
        model = ResNet50Autoencoder().to(device)
        optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()
        loader_kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': True} if device.type == "cuda" else {}

        # 遍历训练集 slide
        for bag_candidate_idx in range(len(bags_dataset)):
            slide_id = str(bags_dataset[bag_candidate_idx]).split(args.slide_ext)[0]
            bag_name = slide_id + '.h5'
            h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
            slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
            output_h5_path = os.path.join(args.data_h5_dir, 'features', f"fold{fold}_{bag_name}")
            output_pt_path = os.path.join(args.data_h5_dir, 'pt', f"fold{fold}_{slide_id}.pt")

            if not args.no_auto_skip and os.path.exists(output_pt_path):
                logging.info(f"Skipping {slide_id} (already processed)")
                continue

            logging.info(f"Processing {slide_id} ({bag_candidate_idx+1}/{len(bags_dataset)}) in Fold {fold}")

            try:
                wsi = openslide.open_slide(slide_file_path)
            except Exception as e:
                logging.error(f"Error opening {slide_file_path}: {e}")
                continue

            dataset = Whole_Slide_Bag_FP(
                file_path=h5_file_path,
                wsi=wsi,
                img_transforms=img_transforms,
                patch_size=args.target_patch_size
            )
            loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

            time_start = time.time()
            avg_loss = train_autoencoder(
                output_path=output_h5_path,
                loader=loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                use_feature_loss=True,
                gradient_clip=1.0,
                slide_id=slide_id,
                data_h5_dir=args.data_h5_dir
            )
            time_elapsed = time.time() - time_start
            logging.info(f"Finished {slide_id} in {time_elapsed:.2f}s, Avg Loss: {avg_loss:.4f}")

            with h5py.File(output_h5_path, "r") as file:
                features = file['features'][:]
            features = torch.from_numpy(features)
            torch.save(features, output_pt_path)

            # 更新调度器
            scheduler.step(avg_loss)

        # 保存折的权重
        weight_path = os.path.join(args.data_h5_dir, f'resnet50_autoencoder_fold{fold}.pth')
        torch.save(model.state_dict(), weight_path)
        logging.info(f"Fold {fold} weights saved to {weight_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet50 Autoencoder for WSI Patches with 10-Fold Training')
    parser.add_argument('--data_h5_dir', type=str, default='/media/public506/sdb/SY/CLAM/CLAM/PATCH')
    parser.add_argument('--data_slide_dir', type=str, default='/media/public506/sdb/SY/349image/image')
    parser.add_argument('--slide_ext', type=str, default='.svs')
    parser.add_argument('--csv_path', type=str, default='/media/public506/sdb/SY/ID.csv')
    parser.add_argument('--splits_dir', type=str, default='/media/public506/sdb/SY/10k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=256)
    parser.add_argument('--use_feature_loss', default=True, action='store_true')
    args = parser.parse_args()
    main(args)