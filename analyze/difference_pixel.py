import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import collections

import json

img1_folder = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/rotated_pred/CLEVR_new_000005.png"
img2_folder = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/rotated/CLEVR_new_000005.png" #pred images

# Initialize a counter for the number of images processed
num_images = 0

with open('/home/qw3971/clevr2/image_generation/camera_rotations_new.json', 'r') as f:
    fixation_data = json.load(f)

# Initialize empty lists to hold all distance and error values
distances = []
errors = []

error_dist = {}

#image_files_1 = [f for f in os.listdir(img1_folder) if os.path.isfile(os.path.join(img1_folder, f))]

# Load the images
image1_path = '/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/rotated_pred/CLEVR_new_000005.png'
image2_path = '/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred/rotated/CLEVR_new_000005.png'
image1 = np.array(Image.open(image1_path).convert('RGB')) / 255.0
image2 = np.array(Image.open(image2_path).convert('RGB')) / 255.0

# Ensure the images are the same size
assert image1.shape == image2.shape, "Images must be the same size"

# Calculate the center of the image
center_x = 160
center_y = 120

# Create a grid of pixel coordinates
Y, X = np.ogrid[:image1.shape[0], :image1.shape[1]]

# Calculate the distance of each pixel from the center
distances = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2).astype(int)

# Compute the squared difference between the images
squared_diff = np.sum((image1 - image2) ** 2, axis=-1)  # Sum over color channels

error_accumulator = collections.defaultdict(list)

for (dist, error) in zip(distances.flatten(), squared_diff.flatten()):
    error_accumulator[dist].append(error)

average_errors = {dist: np.mean(errors) for dist, errors in error_accumulator.items()}

# Flatten the arrays for plotting
distances = np.array(list(average_errors.keys()))
average = np.array(list(average_errors.values()))

# Plot
plt.scatter(distances, average, alpha=0.1)  # Use alpha for better visibility
plt.xlabel('Distance from Center')
plt.ylabel('Squared Difference')
plt.title('Pixel Squared Difference vs. Distance from Center')

plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/rotated/diff.png')