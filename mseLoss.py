import json
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import re
from skimage.metrics import structural_similarity as ssim
import sys
sys.path.append('/scratch/gpfs/qw3971/cnn-vae-old/VAE/')
from RES_VAE_orig import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loading unrotated data
with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/new.json', 'r') as json_file:
    data = json.load(json_file)

orig_keys = list(data.keys())
orig_values = list(data.values())
print(len(orig_keys))

def find_values(data1, data, keys_to_find):
    targets = []
    orig = []
    key_updated = []
    for key in keys_to_find:
        new_key = key.replace(".png", "_rotated.png")
        print(new_key)  
        if new_key in data1:        
            targets.append(data1[new_key])
            key_updated.append(key)
            orig.append(data[key])
        else:
            pass
    return orig, targets, key_updated
    
json_file_path = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/new_rotated.json"
with open(json_file_path, 'r') as json_file:
    data_rotated = json.load(json_file)

orig_data, rotated_data, keys = find_values(data_rotated, data, orig_keys)
print(len(orig_data))

def find_values_in_json_result(json_file, keys_to_find):
    with open(json_file, 'r') as file:
        data = json.load(file)
        results = []

        for key in keys_to_find:
            new_key = key.replace(".png", "_rotated.png")
            print(key+ " " + new_key)
            if new_key in data:
                results.append(data[new_key])

        return results

json_file_path = "/home/qw3971/clevr2/image_generation/combined_file2.json"  

saccades = find_values_in_json_result(json_file_path, keys)
print("check: orig image: {}, rotated image: {}, key_name: {}".format(str(orig_data[0][0]), str(rotated_data[0][0]), str(keys[0])))

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
    
zero = [[0 for _ in sub] for sub in saccades]
#saccades = zero
print("check zero : %s" % str(saccades[0]))
print(len(saccades))

#split into training and test
X_data = list(zip(orig_data, saccades, keys))
y = rotated_data
print(len(X_data))


X_train, X_temp, y_train, y_temp = train_test_split(X_data, y, test_size = 0.1, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.7, random_state = 42)

orig_train, saccade_train, keys_train = zip(*X_train)
orig_val, saccade_val, keys_val = zip(*X_val)
orig_test, saccade_test, keys_test = zip(*X_test)

train_dataset = customDataset(torch.tensor(orig_train), torch.tensor(saccade_train), torch.tensor(y_train), keys_train)
val_dataset = customDataset(torch.tensor(orig_val), torch.tensor(saccade_val), torch.tensor(y_val), keys_val)
test_dataset = customDataset(torch.tensor(orig_test), torch.tensor(saccade_test), torch.tensor(y_test), keys_test)

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle=False) 
val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle=False) 
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle=False) 

#main model
'''
class SaccadeShiftNN(nn.Module):
    def __init__(self):
        super(SaccadeShiftNN, self).__init__()
        self.combine_fc = nn.Linear(4*12*17+3, 64)
        self.relu1 = nn.ReLU()
        self.combine_fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.combine_fc3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.combine_fc4 = nn.Linear(64, 4*12*17)


    def forward(self, latent_input, saccade_input):
        latent_flat = latent_input.view(latent_input.size(0), -1)
        saccade_flat = saccade_input.view(saccade_input.size(0), -1)

        combined = torch.cat((latent_flat, saccade_flat), dim=1)

        combined = self.combine_fc(combined)
        combined = self.relu1(combined)
        combined = self.combine_fc2(combined)
        combined = self.relu2(combined)
        combined = self.combine_fc3(combined)
        combined = self.relu3(combined)
        combined = self.combine_fc4(combined)

        combined = combined.view(-1, 4, 12, 17)

        return combined
 '''
class SaccadeShiftNN(nn.Module):
    def __init__(self):
        super(SaccadeShiftNN, self).__init__()
        self.latent_fc = nn.Sequential(
            nn.Linear(4*12*17, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        '''
        self.saccade_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        '''
        self.combine_fc = nn.Linear(4*12*17, 32)
        self.relu1 = nn.ReLU()
        self.combine_fc2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.combine_fc3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.combine_fc4 = nn.Linear(32, 4*12*17)


    def forward(self, latent_input, saccade_input):
        latent_flat = latent_input.view(latent_input.size(0), -1)
        saccade_flat = saccade_input.view(saccade_input.size(0), -1)

        #combined = torch.cat((latent_flat, saccade_flat), dim=1)

        latent_processed = self.latent_fc(latent_flat)
        #saccade_processed = self.saccade_fc(saccade_flat)

        #combined = torch.cat((latent_processed, saccade_processed), dim=1)

        combined = self.combine_fc(latent_flat)
        combined = self.relu1(combined)
        combined = self.combine_fc2(combined)
        combined = self.relu2(combined)
        #combined = self.combine_fc3(combined)
        #combined = self.relu3(combined)
        combined = self.combine_fc4(combined)

        combined = combined.view(-1, 4, 12, 17)

        return combined 


model = SaccadeShiftNN().to(device)

checkpoint = torch.load("/scratch/gpfs/qw3971/cnn-vae-old/rotated/SO_rotated3/Models/test_run_64.pt", map_location = device)
vae_net = VAE(channel_in = 3, ch=64, latent_channels = 4).to(device)

vae_net.load_state_dict(checkpoint['model_state_dict'])
decoder = vae_net.decoder 

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 200

loss_his = []
loss_val = []

all_outputs = {}
test_latents = {}
train_outputs = {}

for i in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_orig, batch_results, batch_targets, batch_keys in train_dataloader:
        batch_orig = batch_orig.to(device)
        batch_results = batch_results.to(batch_orig.dtype)
        batch_results = batch_results.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()

        saccade_shifted_latent = model(batch_orig, batch_results)

        loss = criterion(saccade_shifted_latent, batch_targets)

        '''

        img_pred = decoder(saccade_shifted_latent)
        img = decoder(batch_targets)

        
        img_pred = torch.clamp(img_pred, 0, 1)
        img = torch.clamp(img, 0, 1) 

        sing_img_pred = img_pred[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()
        sing_img = img[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()

        ssim_loss = 1- ssim(sing_img_pred, sing_img, multichannel = True, channel_axis = -1, data_range = 1.0)


        loss = 0.3 * ssim_loss + 0.7 * mse_loss
        '''

        # all_outputs.extend(saccade_shifted_latent.cpu().detach().numpy().tolist())
        saccade_shifted_latent = saccade_shifted_latent.detach().cpu().numpy()

        for k, latent in zip(batch_keys, saccade_shifted_latent):
            all_outputs[k] = latent.tolist()

        batch_orig = batch_orig.detach().cpu().numpy()

        for k, latent in zip(batch_keys, batch_orig):
            train_outputs[k] = latent.tolist()

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    avg_epoch_loss = train_loss / len(train_dataloader)
    loss_his.append(avg_epoch_loss)
    
    print(f'Epoch [{i+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    # validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_orig, batch_results, batch_targets, _ in val_dataloader:
            batch_orig = batch_orig.to(device)
            batch_results = batch_results.to(device)
            batch_targets = batch_targets.to(device)

            saccade_shifted_latent = model(batch_orig, batch_results)

            loss = criterion(saccade_shifted_latent, batch_targets)

            '''
            img_pred = decoder(saccade_shifted_latent)
            img = decoder(batch_targets)
            
            img_pred = torch.clamp(img_pred, 0, 1)
            img = torch.clamp(img, 0, 1) 
            
            sing_img_pred = img_pred[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()
            sing_img = img[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()
            
            ssim_loss = 1- ssim(sing_img_pred, sing_img, multichannel = True, channel_axis = -1, data_range = 1.0)

            loss = 0.5 * ssim_loss + 0.5 * mse_loss
            '''

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    loss_val.append(avg_val_loss)
    print(f'Epoch [{i+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')

model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_orig, batch_results, batch_targets, batch_keys in test_dataloader:
        batch_orig = batch_orig.to(device)
        batch_results = batch_results.to(device)
        batch_targets = batch_targets.to(device)

        saccade_shifted_latent = model(batch_orig, batch_results)

        loss = criterion(saccade_shifted_latent, batch_targets)

        '''
        img_pred = decoder(saccade_shifted_latent)
        img = decoder(batch_targets)
            
        img_pred = torch.clamp(img_pred, 0, 1)
        img = torch.clamp(img, 0, 1) 
            
        sing_img_pred = img_pred[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()
        sing_img = img[0].detach().cpu().squeeze().permute(1, 2, 0).numpy()
            
        ssim_loss = 1- ssim(sing_img_pred, sing_img, multichannel = True, channel_axis = -1, data_range = 1.0)
    
        loss = 0.5 * ssim_loss + 0.5 * mse_loss
        '''
        
        test_loss += loss.item()

        saccade_shifted_latent = saccade_shifted_latent.detach().cpu().numpy()

        for k, latent in zip(batch_keys, saccade_shifted_latent):
            test_latents[k] = latent.tolist()

avg_test_loss = test_loss / len(test_dataloader)
print(f'Test Loss: {avg_test_loss:.4f}')

with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/test/test_1.json', 'w') as json_file:
    json.dump(test_latents, json_file)
with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/test/train_1.json', 'w') as json_file:
    json.dump(all_outputs, json_file)
with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/test/orig_1.json', 'w') as json_file:
    json.dump(train_outputs, json_file)

plt.figure(figsize=(10, 6))
plt.plot(loss_val)
plt.title('Validation Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Save the figure to a file
plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/rotated/test/test_1.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 6))
plt.plot(loss_his)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Save the figure to a file
plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/rotated/test/train_1.png', dpi=300, bbox_inches='tight')

print('done') 