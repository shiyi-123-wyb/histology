import os 

class Config:
    def __init__(self, input_path: str, selected_input_folders: str, sub_folders: str, output_path: str, process_all: str, batch_size: int, device: str, feature_ext: str,
                 dim_reduce: int, num_distance_groups: int, sample_percentage: float, store_features: bool, store_clusters: bool, 
                 store_plots: bool, store_samples: bool, store_samples_group_wise: bool, use_gpu_clustering: bool):
        
        self.input_path = input_path
        self.selected_input_folders = selected_input_folders
        self.sub_folders = sub_folders 
        self.output_path = output_path
        self.process_all = process_all
        self.batch_size = batch_size
        self.device = device
        self.feature_extractor = feature_ext
        self.dim_reduce = dim_reduce
        self.num_distance_groups = num_distance_groups
        self.sample_percentage = sample_percentage
        self.store_features = store_features
        self.store_clusters = store_clusters
        self.store_plots = store_plots
        self.store_samples = store_samples
        self.store_samples_group_wise = store_samples_group_wise
        self.use_gpu_clustering = use_gpu_clustering  
        
        # Check the input parameters
        print("\n-----------------------------------------------------")
        print("Input/output paths...")
        print("-----------------------------------------------------")
        print("Input_path\t\t:",self.input_path)
        print("Output_path\t\t:",self.output_path)
        print("Feature_extractor\t:",self.feature_extractor) 
        print("Device_used\t\t:",self.device) 
        
        if self.sub_folders: 
            print("Sub_folders\t\t:", self.sub_folders) 
        else:
            print("Sub_folders\t\t: all") 

        if self.process_all:
            print("Processing\t\t: all images in all folders and subfolders") 
            
        if self.store_features:
            self.feature_dir = os.path.join(output_path, 'features')
            os.makedirs(self.feature_dir, exist_ok=True)
            print("Feature_path\t\t:",self.feature_dir)
        else:
            print("Feature_path\t\t: not created (temporary processing)")
            
        if self.store_clusters:
            self.cluster_dir = os.path.join(output_path, 'clusters')
            os.makedirs(self.cluster_dir, exist_ok=True)
            print("Cluster_path\t\t:",self.cluster_dir)
        else:
            print("Cluster_path\t\t: not created")

        if self.store_plots:
            self.plot_dir = os.path.join(output_path, 'plots')
            os.makedirs(self.plot_dir, exist_ok=True)
            print("PCA_plot_path\t\t:",self.plot_dir) 
        else:
            print("PCA_plot_path\t\t: not created") 

        if self.store_samples:       
            self.sample_dir = os.path.join(output_path, 'samples')
            os.makedirs(self.sample_dir, exist_ok=True)
            print("Samples_path\t\t:",self.sample_dir)
            if self.store_samples_group_wise: 
                print("Samples_stored\t\t: group folder wise") 
            else: 
                print("Samples_stored\t\t: cluster folder wise") 
        else:
            print("Samples_path\t\t: not created")
             
        print("-----------------------------------------------------")
        print("Configuration parameters...")
        print("-----------------------------------------------------")
        print("Batch_size\t\t:",self.batch_size)
        print("Sample_percentage\t:",self.sample_percentage)
        print("Num_distance_groups\t:",self.num_distance_groups)
        print("Store_features\t\t:",self.store_features)
        print("Use_GPU_clustering\t:",self.use_gpu_clustering)
        print("-----------------------------------------------------")
        print("Algorithms used...")
        print("-----------------------------------------------------")
        print("Feature extraction\t: AutoEncoder-based encoder")
        print("Feature_size\t\t: 512 ")
        print(f"Feature_reduction\t: PCA (512 -> {self.dim_reduce})")
        print("Clustering_algom\t: K-means")
        print(f"Feature_visualization\t: PCA ({self.dim_reduce} -> 2)")