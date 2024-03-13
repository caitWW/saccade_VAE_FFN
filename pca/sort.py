import json
import os

def count(json_file, white, black):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
            return sum(1 for item in data if item.get('color') == white), sum(1 for item in data if item.get('color') == black)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return 0

# Range of indices for your files
file_indices = range(7000)  # 11 images, from 000000 to 000010

# Base path for the files (update this to your files' directory)
base_path = "/home/qw3971/clevr/image_generation/properties"  # Replace with the actual path

left_img = []
right_img = []

# Loop through each file index
for index in file_indices:
    # Format the filenames
    file_number = str(index).zfill(6)
    json_filename = f"CLEVR_new_{file_number}.json"
    image_filename = f"CLEVR_new_{file_number}.png"
    json_path = os.path.join(base_path, json_filename)

    with open(json_path, 'r') as file:
            data = json.load(file)
    
    left = True
    right = True

    for item in data:
        coord = item.get('pixel_coords')
        if not (coord[0] < 160):
            left = False
        if not (coord[0] > 160):
            right = False  

    # Count occurrences of 'material'
    # white_c, black_c = count(json_path, [0, 0, 0, 1], [1, 1, 1, 1])
    
    if (left == True):
        left_img.append(image_filename)
        print("left", image_filename)
    
    elif (right == True):
        right_img.append(image_filename)
        print("right", image_filename)

print("left:", len(left_img))
print("right:", len(right_img))

# Write the lists to separate JSON files
with open('/scratch/gpfs/qw3971/cnn-vae-old/pca/left.json', 'w') as f:
    json.dump(left_img, f)

with open('/scratch/gpfs/qw3971/cnn-vae-old/pca/right.json', 'w') as f:
    json.dump(right_img, f)

print("done")