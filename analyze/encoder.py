import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
import torch
import sys
from tqdm import trange, tqdm
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
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

root = "/home/qw3971/clevr/image_generation/latent_test/"

png_files = [file for file in os.listdir(root) if file.endswith('.png')]

# heavy cpu load, light memory load
class ImageDiskLoader(torch.utils.data.Dataset):

    def __init__(self, im_ids):
        self.transform = transform
        self.im_ids = im_ids

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        im_path = root + self.im_ids[idx]
        im = Image.open(im_path).convert('RGB')
        #im = crop(im, 30, 0, 178, 178)
        data = self.transform(im)

        return data, self.im_ids[idx]
    
data = ImageDiskLoader(png_files)

kwargs = {'num_workers': 1,
          'pin_memory': True} if use_cuda else {}

data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, **kwargs)

# Access the first element of the dictionary
# values_shifted_tensor = torch.tensor(data_shifted)

checkpoint = torch.load("/scratch/gpfs/qw3971/cnn-vae-old/latent_4/img3/Models/test_run_64.pt", map_location = device)
vae_net = VAE(channel_in = 3, ch=64, latent_channels = 4).to(device)

vae_net.load_state_dict(checkpoint['model_state_dict'])

print('loaded encoder')

latent_vecs = {}



for i, (images, ids) in enumerate(tqdm(data_loader, leave = False)):
     
     recon, mu, log_var = vae_net(images.to(device), ids)
      
     for i in range(len(ids)):
         image_id = ids[i]
         latent_vecs[image_id] = mu[i].cpu().tolist()

with open('/scratch/gpfs/qw3971/cnn-vae-old/analyze/latent_test.json', 'w') as json_file:
    json.dump(latent_vecs, json_file)



print('hi')