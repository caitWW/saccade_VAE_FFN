import json
import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('/scratch/gpfs/qw3971/cnn-vae-old/VAE/')
from RES_VAE_orig import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_cuda = torch.cuda.is_available()
device = torch.device(0 if use_cuda else "cpu") 

# Load the JSON file into a Python dictionary
with open('/scratch/gpfs/qw3971/cnn-vae-old/latent_16/latent.json', 'r') as json_file:
    data = json.load(json_file)

# Access the first element of the dictionary
key = list(data.keys())

values = list(data.values())
values_tensor = torch.tensor(values) 

def find_values_in_json_target(json_file, keys_to_find):
    with open(json_file, 'r') as file:
        data1 = json.load(file)
        targets = []
        orig = []
        key_updated = []

        for key in keys_to_find:
            new_key = "shifted_"+key
            if new_key in data1:
                targets.append(data1[new_key])
                key_updated.append(key)
                orig.append(data[key])
            else:
                pass

        return orig, targets, key_updated
    
json_file_path = "/scratch/gpfs/qw3971/cnn-vae-old/latent_16/latent_shifted.json"
#json_file_path = "/Users/cw/Desktop/latent_vec_shifted.json"
orig, targets, key_updated = find_values_in_json_target(json_file_path, key)
targets_tensor = torch.tensor(targets)
orig_tensor = torch.tensor(orig)

def find_values_in_json_result(json_file, keys_to_find):
    with open(json_file, 'r') as file:
        data = json.load(file)
        results = []

        for key in keys_to_find:
            if key in data:
                results.append(data[key])
            else:
                results[key] = "Key not found in JSON"

        return results



if __name__ == "__main__":
    json_file_path = "/scratch/gpfs/qw3971/cnn-vae-old/run2_saccade.json"  # Replace with the path to your JSON file  # Replace with the list of keys you want to find

    results = find_values_in_json_result(json_file_path, key_updated)
    results_tensor = torch.tensor(results)

import torch
import torch.nn as nn

class customDataset(Dataset):
    def __init__(self, orig_tensor, results_tensor, targets_tensor, keys):
        self.orig_tensor = orig_tensor
        self.results_tensor = results_tensor
        self.targets_tensor = targets_tensor
        self.keys = keys

    def __len__(self):
        return len(self.orig_tensor)

    def __getitem__(self, idx):
        return (self.orig_tensor[idx], self.results_tensor[idx], self.targets_tensor[idx], self.keys[idx])


dataset = customDataset(orig_tensor, results_tensor, targets_tensor, key_updated)
batch_size = 1
num = len(dataset)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True) 

class SaccadeShiftNN(nn.Module):
    def __init__(self):
        super(SaccadeShiftNN, self).__init__()
        self.combine_fc = nn.Linear(16*12*17+2, 128)
        self.combine_fc2 = nn.Linear(128, 64)

        self.output_fc = nn.Linear(64, 128)
        self.output_fc2 = nn.Linear(128, 16*12*17)

    def forward(self, latent_input, saccade_input):
        latent_flat = latent_input.view(latent_input.size(0), -1)
        saccade_flat = saccade_input.view(saccade_input.size(0), -1)

        combined = torch.cat((latent_flat, saccade_flat), dim=1)
        combined_out = F.relu(self.combine_fc(combined))
        combined_out = F.relu(self.combine_fc2(combined_out))

        output = F.relu(self.output_fc(combined_out))
        output = self.output_fc2(output)
        output = output.view(-1, 16, 12, 17)

        return output
        

model = SaccadeShiftNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 100

loss_his = []

all_outputs = []

checkpoint = torch.load("/scratch/gpfs/qw3971/cnn-vae-old/latent_16/img_shifted/Models/test_run_64.pt", map_location = device)
vae_net = VAE(channel_in = 3, ch=64, latent_channels = 16).to(device)

vae_net.load_state_dict(checkpoint['model_state_dict'])
decoder = vae_net.decoder 

all_outputs = {}

for i in range(epochs):
    tot_mse_loss = 0.0
    tot_ssim_loss = 0.0
    tot = 0.0

    optimizer.zero_grad()

    for batch_orig, batch_results, batch_targets, batch_keys in dataloader:
        batch_orig = batch_orig.to(device)
        batch_results = batch_results.to(batch_orig.dtype)
        batch_results = batch_results.to(device)
        batch_targets = batch_targets.to(device)

        saccade_shifted_latent = model(batch_orig, batch_results)

        img_pred = decoder(saccade_shifted_latent)
        img = decoder(batch_targets)

        img_pred = torch.clamp(img_pred, 0, 1)
        img = torch.clamp(img, 0, 1) 

        sing_img_pred = img_pred[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()
        sing_img = img[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()

        ssim_loss = 1- ssim(sing_img_pred, sing_img, multichannel = True, channel_axis = -1, data_range = 1.0)

        mse_loss = F.mse_loss(saccade_shifted_latent, batch_targets)
        loss = 0.5 * ssim_loss + 0.5 * mse_loss

        for k, latent in zip(batch_keys, saccade_shifted_latent):
            all_outputs[k] = latent.tolist()

        loss.backward()

        tot_mse_loss += mse_loss.item()
        tot_ssim_loss += ssim_loss.item()

        tot += loss.item()
        
    optimizer.step()
    optimizer.zero_grad()
    
    avg_mse_loss = tot_mse_loss / len(dataloader)
    avg_ssim_loss = tot_ssim_loss / len(dataloader)
    avg_loss = tot / len(dataloader)
    loss_his.append(avg_loss)
    print(f'Epoch {i+1}, Average MSE Loss: {avg_mse_loss:.4f}, Average SSIM Loss: {avg_ssim_loss:.4f}')

with open('/scratch/gpfs/qw3971/cnn-vae-old/analyze/latent_pred_SL_16.json', 'w') as json_file:
    json.dump(all_outputs, json_file)

plt.figure(figsize=(10, 6))
plt.plot(loss_his)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Save the figure to a file
plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/analyze/loss_SL_16.png', dpi=300, bbox_inches='tight')

print('done') 