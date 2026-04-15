# Standard library imports
import os
from PIL import Image, ImageFile
from PIL import PngImagePlugin
import pandas as pd

# PyTorch imports
from torch.utils.data import Dataset

# Custom module imports
from Config import Config

# tolerate truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True   
# allow very large text chunks
PngImagePlugin.MAX_TEXT_CHUNK = 2**31    
# disable DecompressionBomb protection (optional)
Image.MAX_IMAGE_PIXELS = None            

# Custom Dataset for Whole Slide Images
class WSIDataset(Dataset):
    def __init__(self, input_path, selected_input_folders=None, skip_csv_path=None):
        self.input_path = input_path
        input_folders_list = []
        for folder in os.listdir(input_path):
            folder_path = os.path.join(input_path, folder)
            if os.path.isdir(folder_path):
                input_folders_list.append(folder_path) 
        print(f"🗂️ Number of WSIs (input folders) found: {len(input_folders_list)}")
        print("-----------------------------------------------------")
        for f in input_folders_list:
            print(os.path.basename(f).split(',')[0], end=", ")
        
        # If a CSV is provided with WSI_Name column, read and prepare the set of slides to skip
        skip_slides_set = set()
        if skip_csv_path:
            try:
                skip_df = pd.read_csv(skip_csv_path)
                if 'WSI_Name' in skip_df.columns:
                    skip_slides_set = set(skip_df['WSI_Name'].astype(str).str.strip())
                    print("\n\n🚫 Number of WSIs (input folders) already processed:", len(skip_slides_set)) 
                    print("-----------------------------------------------------") 
                    for f in skip_slides_set:
                        print(f, end=", ")
                else:
                    raise ValueError("CSV file must contain a 'WSI_Name' column.")
            except Exception as e:
                raise ValueError(f"Error reading skip CSV: {e}")
        
        # If specific slides are selected, process only those not in skip_slides_set
        # Otherwise process all slides not in skip_slides_set
        if selected_input_folders and selected_input_folders.strip() and selected_input_folders != "None":  # Check if selected_input_folders is not empty or "None"
            selected_slide_names = set(selected_input_folders.split(','))
            print(f"\n\n✅ Number of selected WSIs (input folders): {len(selected_slide_names)}")
            print("-----------------------------------------------------")  
            for f in selected_slide_names:
                print(f, end=", ")
            # Only process selected slides that are not in the skip list
            self.slide_files = [
                f for f in input_folders_list
                if os.path.basename(f).split(',')[0].strip() in selected_slide_names
                and os.path.basename(f).split(',')[0].strip() not in skip_slides_set
            ]
        else:
            # Process all slides that are not in the skip list
            self.slide_files = [
                f for f in input_folders_list
                if os.path.basename(f).split(',')[0].strip() not in skip_slides_set
            ]
        
        print(f"\n\n📋 Number of WSIs (input folders) to be processed:",len(self.slide_files))
        print("-----------------------------------------------------")
        for f in self.slide_files:
            print(os.path.basename(f).split(',')[0], end=", ")
        print("\n")

        print(f"📊 Summary:")
        print("-----------")
        print(f"  • Total WSIs (input folders) available: {len(input_folders_list)}")
        print(f"  • WSIs (input folders) to skip: {len(skip_slides_set)}")
        print(f"  • WSIs (input folders) to process: {len(self.slide_files)}\n")
        print('-' * 100)
        
    def __len__(self):
        return len(self.slide_files)
    
    def __getitem__(self, idx):
        # This method should be implemented to actually load and process WSI data
        slide_path = self.slide_files[idx]
        # Implement actual slide loading and processing here
        return {"slide_path": slide_path, "slide_name": os.path.basename(slide_path).split(',')[0].strip()}

# Class for handling the images
class TileDataset(Dataset):
    def __init__(self, folder_path: str, config: Config, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = []

        # Determine which folders need to be walked
        self.input_sub_folders = []
        if config.sub_folders is None:
            # If no specific subfolders are given, read images directly from the current folder 
            self.input_sub_folders.append(folder_path)
        else:
            # Parse the comma-separated string into a list of subfolder names 
            sub_folder_names = [name.strip() for name in config.sub_folders.split(',')]
            # Create full paths for each input subfolder
            for name in sub_folder_names:
                subfolder_path = os.path.join(folder_path, name)
                self.input_sub_folders.append(subfolder_path)

        # Walk through each of the determined input folders
        for folder in self.input_sub_folders:
            # Check if the path exists to avoid errors
            if not os.path.isdir(folder):
                #print(f"Warning: Folder '{folder}' does not exist. Skipping.")
                continue
            
            # Collect all image files
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.image_files.append(os.path.join(root, file))   
    
    def __len__(self): 
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy tensor and path instead of recursing
            if self.transform:
                dummy_img = self.transform(Image.new('RGB', (256, 256), color=(0, 0, 0)))
            else:
                dummy_img = Image.new('RGB', (256, 256), color=(0, 0, 0))
            return dummy_img, img_path

def count_images_input_folder(folder_path, config):
    """Count the number of image files in a folder and its subfolders."""
    count = 0
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # Determine which folders need to be walked
    input_folders = []
    if config.sub_folders is None:
        # If no specific subfolders are given, read images directly from the current folder
        input_folders.append(folder_path)
    else:
        # Parse the comma-separated string into a list of subfolder names
        sub_folder_names = [name.strip() for name in config.sub_folders.split(',')]
        # Create full paths for each input subfolder
        for name in sub_folder_names:
            subfolder_path = os.path.join(folder_path, name)
            input_folders.append(subfolder_path)

    # Walk through each of the determined target folders
    for folder in input_folders:
        # Check if the path exists to avoid errors
        if not os.path.isdir(folder):
            print(f"Warning: Folder '{folder}' does not exist. Skipping.")
            continue
 
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    count += 1     
    return count 