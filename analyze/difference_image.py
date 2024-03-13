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


img1_folder = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/rotated"
img2_folder = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/rotated_pred" #pred images

num_images = 0

with open('/home/qw3971/clevr2/image_generation/combined_file2.json', 'r') as f:
    fixation_data = json.load(f)

distances = []
errors = []

error_dist = {}
size_dist = {}

image_files_1 = [f for f in os.listdir(img1_folder) if os.path.isfile(os.path.join(img1_folder, f))]

for image1 in image_files_1:

    parts = image1.split('_')
    if len(parts) > 2 and parts[2].split('.')[0].isdigit():
        image_number = int(parts[2].split('.')[0])
        print(image_number)
        if image_number >= 3500: #first half images only
            continue

    base, ext = os.path.splitext(image1)
    #image2 = image1.replace("_rotated", "")
    image2 = image1
    
    image1_path = os.path.join(img1_folder, image1)
    image2_path = os.path.join(img2_folder, image2)
    print(image1_path)

    if os.path.exists(image1_path) and os.path.exists(image2_path):
        img1 = Image.open(image1_path).convert('RGB')
        img1 = np.array(img1)/255.0
        
        img2 = Image.open(image2_path).convert('RGB')
        img2 = np.array(img2)/255.0

        path = base + "_rotated" + ext
        fix = round(np.abs(fixation_data[path][2]), 2) #round to 2 decimals so only 0.01
        mse = np.mean((img2-img1)**2) #mse for this image
        
        if fix not in error_dist:
            error_dist[fix] = [mse]
            size_dist[fix] = 1
        else:
            error_dist[fix].append(mse)
            size_dist[fix] += 1

average= {}
for fix, mse in error_dist.items():
    average[fix] = sum(mse) / len(mse) #average over this group

dist = np.array(distances)
error = np.array(errors)

# Plot error against distance
plt.scatter(average.keys(), average.values())
#plt.plot(x_axis, y_line, 'r--')
plt.xlabel('Change of Fixation Point in Radians (0-0.7)')
plt.ylabel('Sum of Squared Difference')

# Save the plot
plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/base_line6.png')