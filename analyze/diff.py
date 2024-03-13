import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url

import os,glob
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import trange, tqdm

import json

img1_folder = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/rotated_pred"
img2_folder = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/rotated" #pred images

# Initialize a counter for the number of images processed
num_images = 0

with open('/home/qw3971/clevr2/image_generation/combined_file2.json', 'r') as f:
    fixation_data = json.load(f)

# Initialize empty lists to hold all distance and error values
all_distances = []
all_squared_diffs = []

error_dist = {}

image_files_1 = [f for f in os.listdir(img1_folder) if os.path.isfile(os.path.join(img1_folder, f))]

for image_file in image_files_1:
        image1_path = os.path.join(img1_folder, image_file)
        image2_path = os.path.join(img2_folder, image_file)  # Assuming matching filenames
        
        if os.path.exists(image2_path):
            image1 = np.array(Image.open(image1_path).convert('RGB')) / 255.0
            image2 = np.array(Image.open(image2_path).convert('RGB')) / 255.0
            
            # Calculate distances from the center
            rows, cols = 320, 240
            center_row, center_col = rows / 2, cols / 2
            Y, X = np.ogrid[:rows, :cols]
            distances = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2).astype(int).flatten()
            
            # Calculate squared differences
            squared_diff = np.sum((image1 - image2) ** 2, axis=-1).flatten()
            
            all_distances.extend(distances)
            all_squared_diffs.extend(squared_diff)
# Convert lists to numpy arrays for processing
            
all_distances = np.array(all_distances)
all_squared_diffs = np.array(all_squared_diffs)

unique_distances = np.unique(all_distances)
avg_squared_diffs = [np.mean(all_squared_diffs[all_distances == d]) for d in unique_distances]
    
    # Plot
plt.figure(figsize=(10, 6))
plt.scatter(unique_distances, avg_squared_diffs, alpha=0.6)
plt.xlabel('Distance from Center')
plt.ylabel('Average Squared Difference')
plt.title('Average Pixel Squared Difference vs. Distance from Center Across All Pairs')
plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/difference.png')
