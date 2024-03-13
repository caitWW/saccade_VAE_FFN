import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('/scratch/gpfs/qw3971/cnn-vae-old/VAE/')
from RES_VAE_orig import VAE
import torchvision.utils as vutils

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

# Load the JSON file into a Python dictionary
with open('/scratch/gpfs/qw3971/cnn-vae-old/analyze/latent_test.json', 'r') as json_file:
    latent = json.load(json_file)

l1 = latent["1.png"]
l2 = latent["2.png"]
l3 = latent["0.png"]

l1_tensor = torch.tensor(l1)
l2_tensor = torch.tensor(l2)
l3_tensor = torch.tensor(l3)

'''
flat = diff.view(-1)
n = 10
top_diff = torch.topk(flat, n).indices

def flat_to_multi_indices(flat_indices, shape):
    indices = []
    for idx in flat_indices:
        multi_idx = []
        for dim_size in reversed(shape):
            multi_idx.append(idx % dim_size)
            idx = idx // dim_size
        indices.append(tuple(reversed(multi_idx)))
    return indices

# Convert flat indices to multi-dimensional indices
multi_dim_indices = flat_to_multi_indices(top_diff, diff.shape)

print(l2)

print("diff:", diff)

'''

diff = l2_tensor - l1_tensor 
l4_tensor = l3_tensor + diff
l4 = l4_tensor.tolist()

l = {}

l["CLEVR_new_004880.png"] = l4
l["CLEVR_new_001104.png"] = l1_tensor.tolist()
l["CLEVR_new_005390.png"] = l2_tensor.tolist()
l["CLEVR_new_001359.png"] = l3


with open('/scratch/gpfs/qw3971/cnn-vae-old/analyze/l.json', 'w') as json_file:
    json.dump(l, json_file)

'''
class customDataset(Dataset):
    def __init__(self, keys, tensor):
        self.keys = keys
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return (self.keys[idx], self.tensor[idx])

pred_key = ["hi.png"]
dataset = customDataset(pred_key, l3_tensor)
batch_size = 1
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False) 

checkpoint = torch.load("/scratch/gpfs/qw3971/cnn-vae-old/latent_4/img3/Models/test_run_64.pt", map_location = device)
vae_net = VAE(channel_in = 3, ch=64, latent_channels = 4).to(device)

vae_net.load_state_dict(checkpoint['model_state_dict'])
decoder = vae_net.decoder

print('loaded decoder')

output_folder = '/scratch/gpfs/qw3971/cnn-vae-old/analyze'

l3 = l3_tensor.to(device)
l3 = l3.unsqueeze(0)
pred_img = decoder(l3)

l1 = l1_tensor.to(device)
l1 = l1.unsqueeze(0)
pred_img1 = decoder(l1)

vutils.save_image(
        pred_img1,
        os.path.join(output_folder, "hi.png"),
        normalize=True
    )


for batch_keys, batch_tensor in dataloader:
     batch_tensor = batch_tensor.to(device)

     pred_img = decoder(batch_tensor)
    
    # Save the concatenated image using vutils.save_image
     vutils.save_image(
        pred_img,
        os.path.join(output_folder, f"{batch_keys}"),
        normalize=True
    )

'''