import json
import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the JSON file into dictionary
with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/new.json', 'r') as json_file:
    data = json.load(json_file)

keys = list(data.keys())
values = list(data.values())
values_tensor = torch.tensor(values) 

'''

def modify_key(key):
    # Using regular expression to find and replace the pattern in the key
    new_key = re.sub(r'CLEVR_new_(\d+)_rotated.png', r'CLEVR_new_\1.png', key)
    return new_key

modified_data = {}
for key, value in data.items():
    new_key = modify_key(key)
    modified_data[new_key] = value

# Access the first element of the dictionary
key_final = list(modified_data.keys())
print(key_final)

values = list(modified_data.values())
values_tensor = torch.tensor(values) 
'''

# find each corresponding rotated image
def find_values_in_json_target(json_file, keys_to_find):
    with open(json_file, 'r') as file:
        data1 = json.load(file)
        targets = []
        orig = []
        key_updated = []

        for key in keys_to_find:
            new_key = key.replace(".png", "_rotated.png")
            print(key)
            if new_key in data1:
                
                targets.append(data1[new_key])
                key_updated.append(key)
                #orig.append(modified_data[key])
                orig.append(data[key])
            else:
                pass

        return orig, targets, key_updated
    
json_file_path = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/new_rotated.json"
with open(json_file_path, 'r') as json_file:
    data_rotated = json.load(json_file)
'''
def modify_key(key):
    # Using regular expression to find and replace the pattern in the key
    new_key = re.sub(r'CLEVR_new_(?:rotated)?(\d+)(?:_rotated)?.png', r'CLEVR_new_\1_rotated.png', key)
    return new_key

modified_data_2 = {}
for key, value in data.items():
    new_key = modify_key(key)
    modified_data_2[new_key] = value

path = "/scratch/gpfs/qw3971/cnn-vae-old/rotated/new_rotated2_revised.json"
# Write the modified data back to a new JSON file (optional)
with open(path, 'w') as modified_json_file:
    json.dump(modified_data_2, modified_json_file)
    '''

#json_file_path = "/Users/cw/Desktop/latent_vec_shifted.json"
orig, targets, key_updated = find_values_in_json_target(json_file_path, keys)
targets_tensor = torch.tensor(targets)
orig_tensor = torch.tensor(orig)

print(orig_tensor.shape)
print(targets_tensor.shape)
print(values_tensor.shape)

# find the latent representation of the image from the VAE
def find_values_in_json_result(json_file, keys_to_find):
    with open(json_file, 'r') as file:
        data = json.load(file)
        results = []

        for key in keys_to_find:
            new_key = key.replace(".png", "_rotated.png")
            print(key+ " " + new_key)
            if new_key in data:
                results.append(data[new_key])
            else:
                results[key] = "Key not found in JSON"

        return results



if __name__ == "__main__":
    json_file_path = "/home/qw3971/clevr2/image_generation/combined_file2.json"  # Replace with the path to your JSON file  # Replace with the list of keys you want to find

    results = find_values_in_json_result(json_file_path, keys)
    results_tensor = torch.tensor(results)

import torch
import torch.nn as nn

# organize data into datasets for easy process in NN 
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
batch_size = 32
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True) 

# custom two layer FFN 
class SaccadeShiftNN(nn.Module):
    def __init__(self):
        super(SaccadeShiftNN, self).__init__()
        self.combine_fc = nn.Linear(4*12*17+3, 128)
        self.combine_fc2 = nn.Linear(128, 64)

        self.output_fc1 = nn.Linear(64, 128)
        self.output_fc2 = nn.Linear(128, 4*12*17)

    def forward(self, latent_input, saccade_input):
        latent_flat = latent_input.view(latent_input.size(0), -1)
        saccade_flat = saccade_input.view(saccade_input.size(0), -1)

        combined = torch.cat((latent_flat, saccade_flat), dim=1)
        combined_out = F.relu(self.combine_fc(combined))
        combined_out = F.relu(self.combine_fc2(combined_out))

        output = F.relu(self.output_fc1(combined_out))
        output = self.output_fc2(output)
        output = output.view(-1, 4, 12, 17)

        return output
        

model = SaccadeShiftNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 800

loss_his = []

all_outputs = {}

# start training
for i in range(epochs):
    epoch_loss = 0.0
    for batch_orig, batch_results, batch_targets, batch_keys in dataloader:
        batch_orig = batch_orig.to(device)
        batch_results = batch_results.to(batch_orig.dtype)
        batch_results = batch_results.to(device)
        batch_targets = batch_targets.to(device)

        saccade_shifted_latent = model(batch_orig, batch_results)
        print(saccade_shifted_latent.shape)

        loss = criterion(saccade_shifted_latent, batch_targets)
        # all_outputs.extend(saccade_shifted_latent.cpu().detach().numpy().tolist())
        saccade_shifted_latent = saccade_shifted_latent.detach().cpu().numpy()

        for k, latent in zip(batch_keys, saccade_shifted_latent):
            all_outputs[k] = latent.tolist()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    avg_epoch_loss = epoch_loss / 32
    loss_his.append(avg_epoch_loss)
    
    print(f'Epoch [{i+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/pred_rotated.json', 'w') as json_file:
    json.dump(all_outputs, json_file)

plt.figure(figsize=(10, 6))
plt.plot(loss_his)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# save training loss into graph
plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/rotated/MSE_loss.png', dpi=300, bbox_inches='tight')

print('done') 