import json
import torch
from VAE.RES_VAE import VAE
import os

import torchvision.utils as vutils


use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu")

'''
# Load the JSON file into a Python dictionary
with open('/home/qw3971/cnn-vae-retrain/latent_vec.json', 'r') as json_file:
    data = json.load(json_file)

# Access the first element of the dictionary
key = list(data.keys())

# Access the first element of the dictionary
values_tensor = torch.tensor(list(data.values()))
'''

# Load the JSON file into a Python dictionary
with open('/scratch/gpfs/qw3971/latent_vec_small_test.json', 'r') as json_file:
    data = json.load(json_file)

# Access the first element of the dictionary
key = list(data.keys())

# Access the first element of the dictionary
values_tensor = torch.tensor(list(data.values()))
# values_tensor = torch.tensor(data)

print(values_tensor[0])
print(key[0])
print(data[key[0]])