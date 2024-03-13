import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
import torch
import sys
sys.path.append('/scratch/gpfs/qw3971/cnn-vae-old/VAE/')
from RES_VAE_orig import VAE
from torch.utils.data import DataLoader, Dataset
import os

import torchvision.utils as vutils
import torchvision.transforms as transforms
from PIL import Image

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

transform = transforms.Compose([
    transforms.Pad((0, 8, 0, 8), fill=0, padding_mode='constant'),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])


# Load the JSON file into a Python dictionary
with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred_rotated.json', 'r') as json_file:
    pred_data = json.load(json_file)

keys = []
values = []

# Extract the first four key-value pairs
for i, (key, value) in enumerate(pred_data.items()):
    if i < 4:  # Check to keep only the first four elements
        keys.append(key)
        values.append(value)
    else:
        break
tensor = torch.tensor(values)

pred_key = list(pred_data.keys())

pred_tensor = torch.tensor(list(pred_data.values()))
print(pred_tensor[0].shape)

# Load the JSON file into a Python dictionary
with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/new_rotated.json', 'r') as json_file:
    data = json.load(json_file)
    
# Load the JSON file into a Python dictionary
with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/new.json', 'r') as json_file:
    data2 = json.load(json_file)

print(torch.tensor(list(data.values()))[0].shape)

class customDataset(Dataset):
    def __init__(self, keys, tensor):
        self.keys = keys
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return (self.keys[idx], self.tensor[idx])


dataset = customDataset(pred_key, pred_tensor)
batch_size = 1
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False) 

# Access the first element of the dictionary
# values_shifted_tensor = torch.tensor(data_shifted)

checkpoint = torch.load("/scratch/gpfs/qw3971/cnn-vae-old/rotated/rotated_new/Models/test_run_64.pt", map_location = device)
vae_net = VAE(channel_in = 3, ch=64, latent_channels = 4).to(device)

vae_net.load_state_dict(checkpoint['model_state_dict'])
decoder = vae_net.decoder

checkpoint2 = torch.load("/scratch/gpfs/qw3971/cnn-vae-old/rotated/new/Models/test_run_64.pt", map_location = device)
vae_net2 = VAE(channel_in = 3, ch=64, latent_channels = 4).to(device)

vae_net2.load_state_dict(checkpoint2['model_state_dict'])
decoder2 = vae_net2.decoder

print('loaded decoder')

output_folder = '/scratch/gpfs/qw3971/cnn-vae-old/rotated/rotated_pred'

for batch_keys, batch_tensor in dataloader:
     batch_tensor = batch_tensor.to(device)

     print(batch_tensor.shape)

     pred_img = decoder(batch_tensor)

     
     batch_orig_tensor = [data[key.replace(".png", "_rotated.png")] for key in batch_keys]
     batch_orig_tensor = torch.tensor(batch_orig_tensor).to(device)

     orig_img = decoder(batch_orig_tensor)
     
     batch_orig_tensor2 = [data2[key] for key in batch_keys]
     batch_orig_tensor2 = torch.tensor(batch_orig_tensor2).to(device)

     orig_img2 = decoder2(batch_orig_tensor2)
     
     for j, (pred, orig, orig2) in enumerate(zip(pred_img, orig_img, orig_img2)):

        image_key = batch_keys[j]

        # Debugging: Print shapes of the tensors to confirm they match
        print(f"Shape of pred: {pred.shape}")
        print(f"Shape of orig: {orig.shape}")
        print(f"Shape of orig2: {orig2.shape}")

        print(pred.shape)
        print(orig.shape)

        concatenated_img = torch.cat([orig2, orig, pred], dim = 2)

        # Save the concatenated image using vutils.save_image
        vutils.save_image(
            concatenated_img,
            os.path.join(output_folder, f"{image_key}"),
            normalize=True
        )

print('hi')