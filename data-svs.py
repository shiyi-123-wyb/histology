import torch
import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T

from models.hdmil import KAN_CLAM_MB_v5
from models.hdmil import KAN_CLAM_MB_v4

class Config:
    WSI_PATH_LR = 'path/to/low_res_wsi.svs'   #低分辨率路径
    WSI_PATH_HR = 'path/to/high_res_wsi.svs'  #高分辨率路径
    LR_LEVEL = 2                              #最小层级5
    HR_LEVEL = 0                              #最大层级1

    LIPN_CKPT = 'weights/lipn.pth'            #预训练权重
    DMIN_CKPT = 'weights/dmin.pth'
    PATCH_SIZE_LR = 512                       #16*16
    PATCH_SIZE_HR = 2048                      #256*256
    SCALE_FACTOR = 4                          #低到高分辨率缩放倍数

    MASK_THRESHOLD = 0.5                      #LIPN掩码阈值
    TOP_K_PATCHES = 100                       #DMIN最大处理块

class WSIPreprocessor:
    @staticmethod
    def read_regions(slide_path, level, coords_list, patch_size):
        '''读取指定坐标区域块'''
        slide = openslide.OpenSlide(slide_path)
        patches = []
        for (x,y) in coords_list:
            patch = slide.read_region((x,y), level, (patch_size, patch_size))
            patch = patch.convert('RGB')
            patches.append(patch)
        return patches

    @staticmethod
    def generate_grid_coords(slide, level, patch_size, overlap=64):
        '''生成分块坐标网络'''
        w, h = slide.level_dimensions[level]
        coords = []
        for y in range(0, h, patch_size - overlap):
            for x in range(0, w, patch_size - overlap):
                coords.append((x, y))
        return coords

#def visualize_inference_results(slide, retain_coords_hr, attn_weights):

#---------------------------推理-------------------------
def hdmil_inference():
    lipn = KAN_CLAM_MB_v5().eval().cuda()
    dmin = KAN_CLAM_MB_v4().eval().cuda()
    slide = openslide.OpenSlide(Config.WSI_PATH_HR)
    lipn.load_state_dict(torch.load(Config.LIPN_CKPT))
    dmin.load_state_dict(torch.load(Config.DMIN_CKPT))

    wsi_lr = openslide.OpenSlide(Config.WSI_PATH_LR)
    lr_coords = WSIPreprocessor.generate_grid_coords(wsi_lr, Config.LR_LEVEL, Config.PATCH_SIZE_LR)

    transform = T.Compose([T.Resize(256), T.ToTensor(),#size=16
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    masks = []
    for coord_batch in batch_coords(lr_coords, batch_size=32):
        patches = WSIPreprocessor.read_regions(Config.WSI_PATH_LR, Config.LR_LEVEL,
                                               coord_batch, Config.PATCH_SIZE_LR)
        batch = torch.stack([transform(p) for p in patches]).cuda()
        with torch.no_grad():
            masks.extend(lipn(batch).cpu().numpy())

    #坐标映射
    retain_lr_coords = [lr_coords[i] for i in range(len(masks)) if masks[i] > Config.MASK_THRESHOLD]
    hr_coords = [(x*Config.SCALE_FACTOR, y*Config.SCALE_FACTOR) for (x,y) in retain_lr_coords]

    #DMIN高分辨率推理
    hr_patches = WSIPreprocessor.read_regions(Config.WSI_PATH_HR, Config.HR_LEVEL,
                                              hr_coords, Config.PATCH_SIZE_HR)
    hr_batch = torch.stack([transform(p) for p in hr_patches]).cuda()

    with torch.no_grad():
        features = dmin.feature_extractor(hr_batch)
        attn_weights, _ = dmin.attention(features, features, features)
        selected_idx = torch.topk(attn_weights, k=min(Config.TOP_K_PATCHES, len(features))).indices
        logits = dmin.classifier(features[selected_idx].mean(dim=0))

    prob = torch.softmax(logits, dim=-1)
    print(f'Prediction probabilities: {prob.cpu().numpy()}')
    '''
    attn_weights = attn_weights.cpu().numpy().squeeze()
    visualize_inference_results(
        slide=slide,
        retain_coords_hr=hr_coords,
        attn_weights=(attn_weights - attn_weights.min())/(attn_weights.max() - attn_weights.min())
    )
    '''

def batch_coords(coords, batch_size =32):
    for i in range(0, len(coords), batch_size):
        yield coords[i:i+batch_size]

if __name__ == '__main__':
    hdmil_inference()
