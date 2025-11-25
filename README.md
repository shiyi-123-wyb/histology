 HEAD
# HDMIL
The official implementation of "Fast and Accurate Gigapixel Pathological Image Classification with Hierarchical Distillation Multi-Instance Learning"

## training &  testing
#### step1: CLUSTER/DeepCluster-main
```shell
python sy_train.py \
    --input_path /media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/patches_x20 \
    --output_path /media/joyivan/2/sy/private/TREDENT/20x_256px_0px_overlap/outputss \
    --batch_size 128 \
    --dim_reduce 256 \
    --distance_groups 5 \
    --sample_percentage 0.20 \
    --gpu_ids "0" \
    --test_mode True \
    --batch_size_wsi 10
   
```
#### step2: CLUSTER/DeepCluster-main
```shell
python global_cluster_lsc_pca.py \
    --num_global_clusters 24 \
    --pca_dim 32 \
    --p_landmarks 128 \
    --r_neighbors 5 
```


### for Multiple Classification Tasks
#### step1: CLUSTER
```shell
python baseline.py --pretrain ResNet50_ImageNet --dataset CustomDataset --gpu_id 0 --lr 1e-5 --fold 10 \
    --label_frac 1.0  --n_epochs 100 --wd 1e-5 
```

Updated on 2025年 09月 10日 星期三 21:43:57 CST



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

