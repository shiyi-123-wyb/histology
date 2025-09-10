 HEAD
# HDMIL
The official implementation of "Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning"

## training & validation & testing
### for Binary Classification Tasks
#### step1: DMIN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset Camelyon16 --gpu_id 0 --lr 3e-4 --fold 10 \
    --label_frac 1.00 --degree 12 --init_type xaiver --model v4 --mask_ratio 0.1  
```
#### step2: LIPN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset Camelyon16 --gpu_id 0 --lr 1e-4 --fold 10 \
    --label_frac 1.00 --init_type xaiver --model v5  \
    --pretrain_dir experiments/C10/Res50/init_xaiver/label_frac=1.0/model=v4_degree=12/lr=0.0003_maskratio=0.1/ckpts/ \
    --mask_ratio 0.1  --degree 12 --lwc mbv4t --distill_loss l1 --use_random_inst False 
```


### for Multiple Classification Tasks
#### step1: DMIN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset TCGA-RCC --gpu_id 0 --lr 3e-4 --fold 10 \
    --label_frac 1.00 --degree 12 --init_type xaiver --model v4 --mask_ratio 0.1
```
#### step2: LIPN
```shell
python run.py --pretrain ResNet50_ImageNet --dataset TCGA-RCC --gpu_id 0 --lr 1e-4 --fold 10 \
    --label_frac 1.00 --init_type xaiver --model v5  \
    --pretrain_dir experiments/R10/Res50/init_xaiver/label_frac=1.0/model=v4_degree=12/lr=0.0003_maskratio=0.1/ckpts/ \
    --mask_ratio 0.1  --degree 12 --lwc mbv4t --distill_loss l1 --use_random_inst False
```
Updated on 2025年 09月 10日 星期三 21:43:57 CST

# histology
Multimodal Multi-Instance Analysis of the Relationship between Gene Mutations and Lymph Node Metastasis in Pathology


#省人民医院的数据（结直肠癌）

sy/private/image/------#图像数据

sy/private/labels/GenLyOs_labels.csv/--------#基因，淋巴结，生存分析标签

sy/private/TREDENT/-------#利用trident进行一系列操作的输出


#SurGen 公开数据（结直肠癌）

sy/SurGen/CZI/---------#原始.czi图像数据

sy/SurGen/svs/---------#格式转换后.tiff数据

sy/SurGen/labels/公开标签.xlsx/----------#基因，淋巴结标签

sy/SurGen/TRIDENT/-------#利用trident进行一系列操作的输出


#TCGA-COAD 公开数据（结肠癌）

sy/TCGA-COAD/image/-------#图像数据

sy/TCGA-COAD/clinical.cart.2025-07-09/ --------#临床数据

#TCGA-BRAF  公开数据（乳腺癌）

sy/TCGA-BRAF/image/-------#图像数据

sy/TCGA-BRAF/clinical.cart.2025-07-02/ --------#临床数据

