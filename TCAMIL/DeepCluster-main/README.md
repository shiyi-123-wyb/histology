# DeepCluster++
**STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology**, NeruIPS 2025.  <br>
_Barathi Subramanian, Rathinaraja Jeyaraj, Mitchell Nevin Peterson, Terry Guo, Nigam Shah, Curtis Langlotz, Andrew Y Ng, Jeanne Shen_  
<a href="https://arxiv.org/abs/2511.00383" target="_blank" rel="noopener"> Paper </a> | [Cite](#Citation) 

Modern computer vision projects, across research and industry, often rely on supervised learning, which in turn demands well-curated, diverse training data. To efficiently gather representative samples from large image collections, we introduce DeepCluster++, a semi-automated dataset curation framework with three stages: 

1. extract feature embeddings for all images using a domain-specific encoder (e.g., an autoencoder or a pre-trained backbone) or suitable pretrained encoder; 
2. cluster the embeddings (e.g., with k-means) to group similar samples and then apply equal-frequency binning within clusters to capture diverse patterns for each class; and 
3. have subject-matter experts review the selected samples to confirm label quality.
4. Train a classifier model and validate the model performance.

By tuning a small set of parameters, DeepCluster++ lets us balance the number of samples and the level of diversity, substantially reducing manual effort while yielding high-quality training data for robust models.

<div align="center">
  <img src="https://github.com/rathinaraja/DeepCluster/blob/main/DeepCluster++.jpg" alt="Example" width="950"/>
  <p><em>Figure: DeepCluster++ framework overview</em></p>
</div>

# Typical Workflow - A glance
This example demonstrates how to use DeepCluster++ to curate a diverse training set from tiles extracted out of whole-slide images (WSIs) in digital pathology.
1. Select WSIs that are representative of your cohort (e.g., cases spanning different tissue types).
2. Extract and pre-process tiles (e.g., 256×256 pixels) from each WSI.
3. Preprocess tiles to retain quality tiles.
4. Arrange tiles on the device in a folder structure (single folder, multiple folders, or nested subfolders). DeepCluster++ is designed to work with any specific subfolders or any of these layouts.
5. Feature extraction: Use a domain-specific [pre-trained autoencoder](https://github.com/rathinaraja/AutoEncoder_Image_Reconstruction) or [any other pathology](#Encoders-Available-for-Use)
or real-image foundation model to encode all image tiles in the input directory. [To add new encoder](#Adding-New-Encoder-for-Feature-Extraction) to work with this codebase, it is effortlessly very simple.
6. Clustering: Run k-means on embeddings to group morphologically similar tiles.
7. Diverse sampling: Apply equal-frequency binning (per cluster) to select a balanced, diverse subset for each class.
8. Data collection: Review the samples for each WSI and include them in the appropriate class type.
9. Expert review (optional but recommended): Have a subject-matter expert validate the sampled tiles before finalizing the training set.
10. To train and validate a variety of classifier models, please visit the [STARC-9 Evaluation](https://github.com/rathinaraja/STARC-9-Evaluation) repository. 
    
Although the workflow is demonstrated using WSIs, it is flexible and can be applied to any domain with a collection of images organized in a folder.

**Note:** 
1. Paper link is given here: <a href="https://openreview.net/forum?id=rGWjTlK6Ev" target="_blank" rel="noopener"> Openreview </a>  or <a href="https://arxiv.org/abs/2511.00383" target="_blank" rel="noopener"> Arxiv </a>.
2. If you find our work useful in your research or use parts of this code or dataset, please consider [citing our paper](#Citation).
3. Both the collected dataset and the trained model have been made publicly available for research use. Visit <a href="https://huggingface.co/datasets/Path2AI/STARC-9/tree/main" target="_blank" rel="noopener"> here </a>.

# DeepCluster++ Usage Guide 
We assume representative WSIs have been selected, tiles extracted, and the resulting images filtered using appropriate preprocessing methods. The AutoEncoder (AE) used in this experiement was trained on a set of tiles (images) until the reconstruction quality of test samples become prominent. 

Important to note: 
1. RGB required: Ensure that all the images (tiles) are in RGB format.
2. Encoder input size: The pre-trained autoencoder used here was trained on images of size 256x256 pixels. If your tiles have a different size, either retrain an autoencoder at that size or use a compatible pre-trained encoder to extract features.
3. If you have a single folder with images or multiple folders with images or folders with subfolders or etc. We have designed the program work with any folder structure.
4. Explore the <a href="https://drive.google.com/drive/folders/1pd41-1wAfwGD7XP27OS3KhHAL4xqMXRc" target="_blank" rel="noopener"> input and output folder structure</a> to understand the following instructions.

Input folder structure
-------
Make sure the folder structure is followed as input_folder_1 or input_folder_2. The input folder may either contain images directly (flat structure input_folder_2) or include subfolders (input_folder_1) with images inside as given below. 

Refer to the Test_samples_1 or Test_samples_2 folder to visualize the outcomes of the following executions with various inputs. The command-line arguments can be adjusted based on the folder structure.

<pre>/input_path/Test_samples_1
├── input_folder_1 (WSI_1)
│   ├── sub_folder_1 (sub_folder_1)
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── sub_folder_2 (sub_folder_2)
│   │   └── ...
│   └── sub_folder_m (sub_folder_m)
├── input_folder_2 (WSI_2)
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── input_folder_n (WSI_n)
    ├── image1.png
    ├── image2.png
    └── ...
</pre>

Output folder structure
-------
The following set of files are created in the output folder.
<pre> 
Output
├── clusters
├── features
├── plots
└── samples
└── Summary.csv
</pre>     
<pre> 
clusters (contains clusters of each WSI before sampling)
├── WSI_1
│ ├── Cluster_0
│ ├── Cluster_1
│ └── ...
├── WSI_2
│ └── ...
└── ...
</pre> 
<pre> 
features (2 csv files: features of images and its cluster assignment)
├── WSI_1
│ ├── cluster_assignments.csv
│ └── features.csv
├── WSI_2
│ └── ...
└── ...
</pre>
<pre> 
plots (2 image files: t-SNE visualization with k-means clusters - with and without cluster number)
├── WSI_1
│ ├── tsne_with_legend.png
│ └── tsne_with_numbers.png
├── WSI_2
│ └── ...
└── ...
</pre>
<pre> 
samples (contains samples from the respective clusters)
├── WSI_1
│ ├── Cluster_0
│ ├── Cluster_1
│ ├── Cluster_2
│ └── ...
├── WSI_2
│ └── ...
└── ...
</pre>

System requirements
-----------------------
### Minimum hardware requirements
+ RAM: 4 GB
+ Processor: Intel i5/i7 (or AMD Ryzen 5/7 equivalent)
+ Storage: 512 GB  
+ GPU: Optional (possible to run on CPU)

### Recommended hardware requirements for faster execution
+ RAM: 8 GB
+ Processor: Intel i7-12th gen or newer (or AMD Ryzen 7 5000 series+)
+ Storage: 1 TB SSD (NVMe preferred)
+ GPU: NVIDIA RTX 4060 or better (16GB+ VRAM) 

Create a virtual environment and install the required packages
-------
```bash
conda env create -f environment.yml
conda activate DeepCluster++
``` 
Command-line arguments 
-------
+ Each input folder (WSI) should include a minimum of 256 images to match the 256 PCA components used. Folder with less than 256 images is still valid but there is no DeepCluster++ applied.
+ Consider the following key details about Test_samples to better understand the command-line arguments.
    - Input_path - /path/Test_samples_1
    - Input folders (WSIs) - WSI_1, WSI_2, WSI_3, WSI_4, and WSI_5
    - Sub folders (if available) - sub_folder_1, sub_folder_2, sub_folder_3, sub_folder_4
+ Number of cluster for each input folder is determined by taking square root of number of samples in each input folder. 

| Argument                   | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `--input_path /path/Test_samples_1`      | The input path containing a set of input folders, each corresponding to a WSI. |
| `--selected_input_folders "WSI_1,WSI_2"`       | Process specific input folders in the path (e.g., WSI names). By default, if not passed, all input folders in the path are considered. |
| `--sub_folders "sub_folder_1,sub_folder_2"`  | If an input folder contains subfolders, specify which ones to process. By default, all subfolders in the input folder are considered. |
| `--process_all True`                  | Process all the images in the given input path regardless of input folders and sub_folders. |
| `--output_path /path/Test_samples_1_output`          | Output path to store extracted features, clusters, plots, and samples. |
| `--feature_ext encoder_name`          | Encoder name to extract features. |
| `--device cpu`                       | Optional. Default: None. Device type (cpu or all_gpus). |
| `--gpu_ids 4,5`                          | Optional. Default: GPU `0` is assigned. |
| `--use_gpu_clustering`                   | Optional. Uses GPU for clustering (requires RAPIDS cuML, default: False) |
| `--batch_size 128`                     | Optional. Default: `128`. Recommended: `256`. |
| `--dim_reduce 256`                     | Optional. Default: `256`. Speicify the dimensionality reduction size. |
| `--distance_groups 5`                 | Default: `5`. Dividing the cluster into 5 groups.|
| `--sample_percentage 0.2`             | Default: `0.2` (sample 20% in a cluster). Increase this value to collect more samples. |
| `--model AE_CRC.pth`                  | Model path. By default, `AE_CRC.pth` in the current path is used. |
| `--seed 42`                           | Optional. Default: `42`. |
| `--store_features True`               | Stores the features of input folders. Default: `False` (features are not stored). |
| `--store_clusters True`               | Stores the clusters. Default: `False` (clusters are not stored). |
| `--store_plots True`                  | Stores the plots of clusters. Default: `False` (plots are not stored). |
| `--store_samples True`                | Stores the samples of clusters. Default: `False` (samples are not stored). |
| `--store_samples_group_wise True`     | Stores the samples group-wise for each cluster. Default: `False` (samples are not stored). |

Command-line Usage
-------  
### Basic Run
The parameters input_path, output_path, and feature_ext are mandatory and must be specified with every program run.

To process all the input folders (WSIs) independently in the input path regardless of subfolders or images in each input folder. 

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc
```
> **Note:** The above execution does not store any output data by default as the default argument is set to False.

### Available Encoders

Available feature extractor names in the project are ae_crc, resnet50, resnet50_1024, resnet18, densenet121, efficientnet_b0, efficientnet_b7, vit_b16, custom_cnn, uni, conch, prov_gigapath, and ctranspath.

| Category                   | Encoders                  |
|-----------------------------|---------------------------|
| ResNet family      | resnet18, resnet50, resnet50_1024 |
| DenseNet family       | densenet121 |
| EfficientNet family  | efficientnet_b0, efficientnet_b7 |
| Vision Transformers   | vit_b16, ctranspath |
| Specialized encoders  | custom_cnn, uni, conch, prov_gigapath |

Instructions are given below to add new encoders to extract features from.

### Storing Results

To store extracted features, clusters, samples, and cluster plots, pass `True` for the appropriate command line arguments: 

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --store_features True --store_clusters True --store_plots True --store_samples True
```

### Logging Output

To log all print statements into a text file, append `| tee output.txt` at the end of your command in the terminal:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --store_features True --store_clusters True --store_plots True --store_samples True | tee Sample_output.txt
``` 
The samle output displayed on the commandline is shown [here](https://github.com/rathinaraja/DeepCluster/blob/main/Sample_output.txt).

### Processing Images Directly

If the input path contains only images (no input), those images will be processed directly. **Ensure at least 256 images are present in the input folder:**

```bash
python Main.py --input_path /path/Test_samples_1/WSI_2 --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --store_features True --store_clusters True --store_plots True --store_samples True
```
To process all the images from all sub folders in the input folder

```bash
python Main.py --input_path /path/Test_samples_1/WSI_1 --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --process_all True --store_features True --store_clusters True --store_plots True --store_samples True
```

## Advanced Usage

### Processing Specific Input Folders

To process specific input folders (e.g., WSI_1 and WSI_4), use the `--selected_input_folders` parameter:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --selected_input_folders "WSI_1,WSI_4" --store_features True --store_clusters True --store_plots True --store_samples True
```

### Storing Specific Results

#### Store Only Clusters
```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --store_clusters True
```

#### Store Only Samples
```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --store_samples True
```

### Processing Specific Subfolders

To process specific subfolders (e.g., "sub_folder_1, sub_folder_3") within the folders in the input path, use the `--sub_folders` parameter:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --sub_folders "sub_folder_1,sub_folder_3" --store_clusters True --store_samples True
```

### Combined Folder and Subfolder Selection

To process specific subfolders (e.g., "sub_folder_1, sub_folder_3") within specific input folders (e.g., WSI_1 and WSI_3), use both `--selected_input_folders` and `--sub_folders` parameters:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output --feature_ext ae_crc --selected_input_folders "WSI_1,WSI_3" --sub_folders "sub_folder_1,sub_folder_3" --store_clusters True --store_samples True
```

### Sample Organization Options

By default, with `--store_samples True`, samples are stored by cluster. To store samples in group folders within clusters:

```bash
python Main.py --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output --feature_ext ae_crc --store_samples_group_wise True
```

### CPU and GPU usage

By default, execution runs on the CPU. To process input folders/WSIs serially using CPU only, we can set CPU option explicitly.

```bash
python Main.py  --input_path /path/Test_samples/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --selected_input_folders "WSI_1,WSI_3" --device cpu --store_clusters True --store_samples True 
```

To utilize a specific GPU ID or enable all available GPUs, configure the settings accordingly as follows.

#### All Available GPUs Processing
Process input folders/WSIs in parallel using all available GPUs:
```bash
python Main.py  --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --selected_input_folders "WSI_1,WSI_3" --device all_gpus --store_features True --store_clusters True --store_plots True --store_samples True 
```
If there are "n" input folders, then based on the number of GPUs available in machine, "n" distributed equally.

#### Single GPU Processing

Process input folders/WSIs serially using a single GPU:

```bash
python Main.py  --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output --feature_ext ae_crc --selected_input_folders "WSI_1,WSI_3" --device all_gpus --gpu_ids 0 --store_clusters True --store_samples True 
```
#### Multiple GPU Processing

Process input folders/WSIs in parallel using multiple specified GPUs:

```bash
python Main.py  --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --selected_input_folders "WSI_1,WSI_3" --device all_gpus --gpu_ids 4,5 --store_clusters True --store_samples True 
```

#### GPU accelaration for clustering and sampling process
K-means and sampling process are executed by CPU. If you run the program on powerful GPUs, when there 1000s of samples, it is good to make use of them to accelarate clustering and samplingunnig on GPUs:
```bash
python Main.py  --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext ae_crc --selected_input_folders "WSI_1,WSI_3" --device all_gpus --use_gpu_clustering True --store_clusters True --store_samples True 
```

### Sample Output with Multi-GPU WSI Processing Pipeline

Go to the respective folders to view the **sample input** (<a href="https://drive.google.com/file/d/153yQcZEtRHr8Cxj78tH-pCHYj0pZ1ur3/view?usp=sharing" target="_blank" rel="noopener">/path/Test_samples_1/</a>) and **generated output** (<a href="https://drive.google.com/file/d/1wgqlqieBn0hNS5FPr6R2BP8aSc4KNQEj/view?usp=sharing" target="_blank" rel="noopener">/path/Output/</a>) produced by the pipeline.

The <a href="https://github.com/rathinaraja/DeepCluster/blob/main/Summary.csv" target="_blank" rel="noopener">summary.csv</a> file provides key statistics including the number of clusters, number of samples selected, and timing information for each WSI processed.

| **WSI_Name** | **Total_images** | **Num_clusters** | **Num_samples_selected** | **Feature_extraction_time (s)** | **Clustering_time (s)** | **Sampling_time (s)** | **Overall_time (s)** | **Date_time_processed** | **Additional_note** |
|:--------------|----------------:|-----------------:|-------------------------:|--------------------------------:|------------------------:|----------------------:|----------------------:|--------------------------|----------------------|
| WSI_5 | 500 | 23 | 109 | 14.72 | 4.32 | 0.33 | 19.37 | 2025-10-23 16:37:13 PDT | GPU:1 |
| WSI_1 | 600 | 25 | 132 | 14.21 | 4.97 | 0.39 | 19.58 | 2025-10-23 16:37:13 PDT | GPU:3 |
| WSI_2 | 600 | 25 | 130 | 13.28 | 5.94 | 0.42 | 19.64 | 2025-10-23 16:37:13 PDT | GPU:2 |
| WSI_3 | 600 | 25 | 144 | 14.30 | 4.74 | 0.40 | 19.43 | 2025-10-23 16:37:13 PDT | GPU:0 |
| WSI_4 | 500 | 23 | 119 | 13.86 | 4.90 | 0.35 | 19.12 | 2025-10-23 16:37:13 PDT | GPU:4 |

- The pipeline distributes workload dynamically across multiple GPUs for parallel WSI processing.  
- Each stage (feature extraction, clustering, sampling) is timed and recorded in `Summary.csv`.  
- All results, plots, and cluster visualizations are saved in the specified output directory.

## Parameters Reference

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--input_path` | Path to input directory containing WSIs | Required | `/path/Test_samples_1/` |
| `--output_path` | Path to output directory | Required | `/path/Test_samples_1_output/` |
| `--feature_ext ` | Encoder name to extract features | Required | `resnet50, resnet50_1024, resnet18, densenet121, efficientnet_b0, efficientnet_b7, vit_b16, custom_cnn, uni, conch, prov_gigapath, and ctranspath` |
| `--selected_input_folders` | Comma-separated list of specific folders to process | `None` (all folders) | `"WSI_1,WSI_4"` |
| `--sub_folders` | Comma-separated list of specific subfolders to process | `None` (all subfolders) | `"sub_folder_1,sub_folder_3"` |
| `--process_all` | Process all the images in the input path | `None` | `True` |
| `--gpu-ids` | Specify the GPU ID or list of GPU IDs | `None` | `4,7` |
| `--device` | Process with CPU or all GPUs in the system | `cpu` | `cpu/all_gpus` |
| `--use_gpu_clustering` | Accelarate clustering and sampling using GPUs | `False` | `True` |
| `--batch_size` | Batch size to work on GPU | `64` | `Recommended to set 256` |
| `--dim_reduce` | Reducing feature size from encoder before clustering | `256` | `Recommended to set 256` |
| `--distance_groups` | Ensuring the diversty of images within a cluster | `5` | `Increase/decrease based on the requirement` |
| `--model` | Path to the encoder | `5` | `Increase/decrease based on the requirement` |
| `--sample_percentage` | Sampling images from clusters  | `AE_CRC.pth` | `Current path` |
| `--store_features` | Store extracted features | `False` | `True`/`False` |
| `--store_clusters` | Store cluster results | `False` | `True`/`False` |
| `--store_plots` | Store visualization plots | `False` | `True`/`False` |
| `--store_samples` | Store sample images | `False` | `True`/`False` |
| `--store_samples_group_wise` | Organize samples by groups instead of clusters | `False` | `True`/`False` |

## Requirements

- Minimum 256 images per input folder for effective clustering
- Supported image formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`

# Encoders Available for Use

| Encoder Name | Invocation Name | Architecture | Feature Dim | Use Case | Model Size | Memory | Parameters | Notes |
|---------|----------|--------------|------------|----------|-----------|--------|-----------|-------|
| AE_CRC | `ae_crc` | Custom AutoEncoder | 512 | Domain specific | Medium | ~2 GB | 13 M | Lightweight, custom architecture for histopathology |
| ResNet50 | `resnet50` | ImageNet-pretrained CNN | 2048 | Standard Baseline | Large | ~4 GB | 24 M | Industry standard, reproducible baseline |
| ResNet50_1024 | `resnet50_1024` | ImageNet-pretrained CNN (Layer3) | 1024 | Efficient Baseline | Large | ~3 GB | 24 M | ResNet50 with intermediate layer extraction |
| ResNet18 | `resnet18` | ImageNet-pretrained CNN | 512 | Lightweight Baseline | Small | ~2 GB | 11.7 M | Computationally efficient alternative to ResNet50 |
| DenseNet121 | `densenet121` | ImageNet-pretrained Dense CNN | 1024 | General Purpose | Medium | ~4 GB | 8 M | Dense connections, parameter efficient |
| EfficientNetB0 | `efficientnet_b0` | EfficientNet Family | 1280 | Efficient Extraction | Small | ~2 GB | 5.3 M | Mobile-friendly, optimal efficiency-accuracy tradeoff |
| EfficientNetB7 | `efficientnet_b7` | EfficientNet Family | 2560 | Maximum Efficiency | Very Large | ~8 GB | 64 M | Largest EfficientNet, best performance |
| ViT_B16 | `vit_b16` | Vision Transformer | 768 | Transformer-based | Large | ~6 GB | 86 M | Self-attention mechanism, global receptive field |
| CustomCNN | `custom_cnn` | Custom CNN Architecture | Variable | Experimentation | Medium | ~4 GB | 3.9 M | User-defined architecture for custom tasks |
| UNI | `uni` | Pathology Foundation Model | 1024 | Histopathology-specific | Large | ~18 GB | 88 M | State-of-the-art for medical imaging, heavyweight |
| CONCH | `conch` | Contrastive Learning Model | 512 | Histopathology-specific | Very Large | ~20 GB | 87 M | Contrastive pre-training, excellent for WSI |
| Prov-GigaPath | `prov_gigapath` | Pathology Foundation Model | 1536 | Histopathology-specific | Very Large | ~22 GB | 305 M | Largest pathology model, cutting-edge features |
| CTransPath | `ctranspath` | Transformers for Pathology | 768 | Histopathology-specific | Large | ~8 GB | 87 M | Lightweight pathology transformer |

## Encoder Categories

### 1. Lightweight Baselines
- **`ae_crc`, `resnet18`, `efficientnet_b0`** - Best for quick experimentation, limited GPU memory
- Memory: 2-4 GB | Parameters: 5.3M-13M | Features: 512-1280

### 2. Standard Baselines
- **`resnet50`, `resnet50_1024`, `densenet121`** - Reproducible, well-established
- Memory: 3-4 GB | Parameters: 8M-25.6M | Features: 512-2048

### 3. Advanced Architectures
- **`efficientnet_b7`, `vit_b16`** - State-of-the-art general-purpose models
- Memory: 6-8 GB | Parameters: 66.3M-86M | Features: 768-2560

### 4. Pathology-Specific Foundation Models
- **`uni`, `conch`, `prov_gigapath`, `ctranspath`** - Specialized for histopathology/WSI
- Memory: 8-22 GB | Parameters: 55.2M-435M | Features: 512-1536
- **Recommended for whole slide imaging and computational pathology**

## Recommended Configurations

### For Classification Tasks
| Scenario | Primary | Secondary | Tertiary |
|----------|---------|-----------|----------|
| Maximum Accuracy | `prov_gigapath` | `uni` | `conch` |
| Balanced Performance | `uni` | `ctranspath` | `resnet50` |
| GPU Memory Constrained | `ctranspath` | `resnet50` | `densenet121` |
| Quick Prototyping | `resnet50` | `densenet121` | `resnet18` |
| Baseline Comparison | `resnet50` | `resnet50_1024` | `densenet121` |

### Feature Dimension Impact

- **512 dims**: `ae_crc`, `resnet18`, `conch` - Lower dimensional, faster downstream processing
- **768 dims**: `vit_b16`, `ctranspath` - Balanced representation
- **1024 dims**: `densenet121`, `uni`, `resnet50_1024` - Standard medical imaging choice
- **1280 dims**: `efficientnet_b0` - Efficient dense representation
- **1536 dims**: `prov_gigapath` - Rich feature representation
- **2048 dims**: `resnet50` - High-dimensional baseline
- **2560 dims**: `efficientnet_b7` - Maximum dimensional features

### Memory-Performance Tradeoff

**High-Performance (Accept High Memory)**
```
prov_gigapath (22GB) > uni (18GB) > conch (20GB) > efficientnet_b7 (8GB) > ae_crc (2GB)
```
**Balanced Approach**
```
ctranspath (8GB) > vit_b16 (6GB) > resnet50 (4GB) > ae_crc (2GB)
```
**Memory-Efficient**
```
ae_crc (2GB) > resnet18 (2GB) > efficientnet_b0 (2GB) > resnet50_1024 (3GB) 
```
### Quick Selection Guide

- **Best Overall for Pathology**: `prov_gigapath` or `uni`
- **Best Baseline**: `resnet50`
- **Best If GPU-Limited**: `ctranspath` or `resnet50`
- **Best for Speed**: `resnet18` or `efficientnet_b0`
- **Best for Experimentation**: `ae_crc` or `densenet121`
- **Best Transformer Option**: `vit_b16` or `ctranspath`

## Adding New Encoder for Feature Extraction

The framework is highly modular, enabling seamless integration of new encoders. This design ensures maximum flexibility, scalability, and reproducibility across diverse computational pathology workflows. Follow these **4 simple steps** to add a new encoder:

### 1. Add the Model File

Place a new encoder file (e.g., `NewEncoder.py`) in the `Encoders/` directory.

### 2. Import the Encoder

Add the following line to import the encoder in `Feature_extraction.py` file:

```python
from Encoders.NewEncoder import NewEncoder
```

### 3. Register the Encoder

Add it to the encoder dictionary in `Feature_extraction.py`:

```python
Available_Encoders = {
    'new_en': NewEncoder,
    ...
}
```

### 4. Use the New Encoder

Specify the encoder name in `Config.py`:

```python
python Main.py  --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext new_en --selected_input_folders "WSI_1,WSI_3" --device all_gpus --use_gpu_clustering True --store_clusters True --store_samples 
```

### Benefits

This modular structure ensures:
- ✅ Maximum flexibility
- ✅ Scalability
- ✅ Reproducibility
- ✅ Easy integration of diverse encoders across computational pathology workflows

## HuggingFace Token Setup for Pathology Encoders

Models such as **UNI**, **CONCH**, **Prov-GigaPath**, and **CTransPath** require permission to download from HuggingFace. Follow these steps to set up authentication.

### Step 1: Request HuggingFace Tokens

Visit [HuggingFace Settings - Tokens](https://huggingface.co/settings/tokens) to generate access tokens for the following models:

- [UNI](https://huggingface.co/MahmoodLab/UNI)
- [CONCH](https://huggingface.co/MahmoodLab/CONCH)
- [Prov-GigaPath](https://huggingface.co/prov-gigapath/Prov-GigaPath)
- [CTransPath](https://huggingface.co/Xiyue/CTransPath)

Create a **Fine-grained** token with `repo.content.read` permissions.

---

### Step 2: Install Required Libraries

If not already installed, install the necessary packages:

```bash
pip install timm
conda install -c conda-forge huggingface_hub
pip install -U huggingface_hub
```

---

### Step 3: Authenticate with HuggingFace

Run the following command to log in:

```bash
hf_auth login
```

You will see the HuggingFace banner:

```
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
```

### Step 3a: Enter Your Token

When prompted, enter your HuggingFace token (input will not be visible):

```
To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible):
```

Paste the token you generated from HuggingFace.

### Step 3b: Save Token as Git Credential

When asked to add token as git credential, respond with `Y`:

```
Add token as git credential? (Y/n) y
```

The system will confirm:

```
Token is valid (permission: fineGrained).
The token has been saved to /home/user/.cache/huggingface/stored_tokens
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/user/.cache/huggingface/token
Login successful.
```

---

### Step 4: Configure Git Credential Helper (Optional)

To ensure credentials are properly stored, configure git:

```bash
git config --global credential.helper store
```

---

## Verification

Verify your authentication:

```bash
hf auth whoami
```

You should see output indicating your logged-in token and permissions.

To log out if needed:

```bash
hf auth logout
```

---

### Summary

| Step | Command |
|------|---------|
| Install packages | `pip install timm && conda install -c conda-forge huggingface_hub && pip install -U huggingface_hub` |
| Authenticate | `hf auth login` |
| Store credentials | `git config --global credential.helper store` |
| Verify login | `hf auth whoami` |
| Logout (if needed) | `hf auth logout` |

---

### Troubleshooting

#### Issue: Permission Denied for Model

**Solution**: Ensure you have accepted the model's license on HuggingFace and your token has `repo.content.read` permissions.

#### Issue: Token Already Saved

If you see `A token is already saved on your machine`, you can:

- View current token: `hf auth whoami`
- Replace with new token: `hf auth login` (will erase existing token)
- Logout and login: `hf auth logout && hf auth login`

#### Issue: Models Not Downloading

**Solution**: Verify authentication is active:

```bash
hf auth whoami
```

Then try downloading a model again.

---

### Next Steps

After authentication, you can use these encoders in your pipeline: For example, to use uni encoder,

```python
python Main.py  --input_path /path/Test_samples_1/ --output_path /path/Test_samples_1_output/ --feature_ext uni --selected_input_folders "WSI_1,WSI_3" --device all_gpus --use_gpu_clustering True --store_clusters True --store_samples 
```

The encoders will automatically download the models using your authenticated token.


# Explore Clusters and Collect Representative Tiles

## Overview
Explore the different clusters of each WSI to collect representative tiles across the following tissue types in our project:

| Tissue Type | Abbreviation | Description |
|-------------|--------------|-------------|
| **ADI** | Adipose | Adipose tissue/Fat cells |
| **LYM** | Lymphocyte | Lymphocytic infiltration |
| **MUS** | Muscle | Muscle tissue |
| **FCT** | Fibrous Connective Tissue | Loose connective tissue |
| **MUC** | Mucin | Mucin-rich areas |
| **NCS** | Necrotic Debris | Necrotic/Dead tissue |
| **BLD** | Blood | Red blood cells/Vascular areas |
| **TUM** | Tumor | Tumor tissue |
| **NOR** | Normal | Normal epithelial tissue |

## Cluster Exploration Process

**Review Generated Clusters**
   ```bash
   # Navigate to cluster output directory
   cd /output_path/clusters/
   
   # Review clusters for each WSI
   ls -la WSI_*/Cluster_*
   ```
**Visual Inspection** 
   - Review sample images in each sample folder
   - Identify clusters that predominantly contain specific tissue types
     
**Manual Curation**
   - Review sampled tiles from each cluster
   - Create tissue-type specific folders:
     ```
     representative_tiles/
     ├── ADI/
     ├── LYM/
     ├── MUS/
     ├── FCT/
     ├── MUC/
     ├── NCS/
     ├── BLD/
     ├── TUM/
     └── NOR/
     ```

## Quality Assurance
- Ensure balanced representation across all 9 tissue types
- Verify cluster purity for each tissue type
- Document cluster-to-tissue-type mapping for reproducibility

# Pathologist Verification

## Training Set Validation

The collected representative tiles undergo rigorous pathologist verification to ensure:

### Annotation Quality Control
- **Expert Review**: Board-certified pathologists examine each representative tile
- **Consensus Building**: Multiple pathologists review ambiguous cases
- **Documentation**: All annotations are documented with reasoning

### Verification Process
```bash
# Organize tiles for pathologist review
mkdir -p pathologist_review/{pending,verified,rejected}

# Move tiles to pending review folder
cp -r representative_tiles/* pathologist_review/pending/
```

### Verification Criteria
| Criteria | Description | Action |
|----------|-------------|---------|
| **Tissue Type Accuracy** | Correct classification of tissue type | Accept/Reject/Reclassify |
| **Image Quality** | Clear, well-stained, artifact-free | Accept/Reject |
| **Representative Nature** | Typical example of tissue type | Accept/Request alternatives |
| **Diagnostic Relevance** | Clinically relevant features present | Accept/Enhance dataset |

### Verification Workflow
**Initial Review**: Pathologist examines tiles by tissue type

**Quality Assessment**: Rate each tile (1-5 scale)

**Consensus Meeting**: Resolve disagreements

**Final Dataset**: Create verified training set

# Classifier Model Training and Evaluation
To explore model benchmarking and performance validation on the STARC-9 dataset, refer to the companion repository [STARC-9 Evaluation](https://github.com/rathinaraja/STARC-9-Evaluation). This repository provides scripts and workflows for training and validating a variety of classifier models, including baseline and pathology specific foundation architectures. It also includes evaluation metrics, visualization tools, and reproducible experiment setups for comparative analysis.

```bash
# After verification, organize final training set
mkdir -p verified_training_set/{ADI,LYM,MUS,FCT,MUC,NCS,BLD,TUM,NOR}

# Move verified tiles to final training folders
# (based on pathologist recommendations)
```

# License & Support
For issues, questions, or feature requests:
+ This repository is made available under the CC BY-NC 4.0 License and is available for non-commercial academic purposes.
+ Open an issue on GitHub
+ Contact: jrathinaraja@gmail.com
+ Website: https://jrathinaraja.co.in/contact/

# Funding
Funding for this study was provided by the United States National Cancer Institute (NCI), National Institutes of Health (NIH) (R01 CA270437).

# Citation
If you find our work useful in your research or use parts of this code or dataset, please consider citing our paper <a href="https://openreview.net/forum?id=rGWjTlK6Ev" target="_blank" rel="noopener"> Openreview </a>  or <a href="https://arxiv.org/abs/2511.00383" target="_blank" rel="noopener"> Arxiv </a>.  

APA 6
```bash
Subramanian, B., Jeyaraj, R., Peterson, M. N., Guo, T., Shah, N., Langlotz, C., Ng, A. Y., & Shen, J. (2025). STARC-9: A large-scale dataset for multi-class tissue classification for CRC histopathology. In The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track. https://openreview.net/forum?id=rGWjTlK6Ev
```

BibTex

```bash
@inproceedings{
subramanian2025starc,
title={{STARC}-9: A Large-scale Dataset for Multi-Class Tissue Classification for {CRC} Histopathology},
author={Barathi Subramanian and Rathinaraja Jeyaraj and Mitchell Nevin Peterson and Terry Guo and Nigam Shah and Curtis Langlotz and Andrew Y. Ng and Jeanne Shen},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=rGWjTlK6Ev}
}
```
