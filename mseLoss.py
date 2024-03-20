import json
import torch 
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the JSON file into a Python dictionary
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

#split into training and test
X_data = list(zip(orig_tensor, results_tensor, key_updated))
y = targets_tensor
X_train, X_temp, y_train, y_temp, keys_train, keys_temp = train_test_split(X_data, y, key_updated, test_size = 0.1, random_state = 42)
X_val, X_test, y_val, y_test, keys_val, keys_test = train_test_split(X_temp, y_temp, keys_temp, test_size = 0.9, random_state = 42)

# Unpack the training, validation, and test sets
orig_train, results_train, keys_train = zip(*X_train)
print("results length orig " + str(len(results_train)))
orig_val, results_val, keys_val = zip(*X_val)
orig_test, results_test, keys_test = zip(*X_test)

# Convert back to tensors
orig_train_array = np.stack([np.array(item) for item in orig_train])
orig_train_tensor = torch.tensor(orig_train_array)

results_train_array = np.stack([np.array(item) for item in results_train])
results_train_tensor = torch.tensor(results_train_array)

targets_train_array = np.stack([np.array(item) for item in y_train])
targets_train_tensor = torch.tensor(targets_train_array)

orig_val_array = np.stack([np.array(item) for item in orig_val])
orig_val_tensor = torch.tensor(orig_val_array)
results_val_array = np.stack([np.array(item) for item in results_val])
results_val_tensor = torch.tensor(results_val_array)
targets_val_array = np.stack([np.array(item) for item in y_val])
targets_val_tensor = torch.tensor(targets_val_array)

orig_test_array = np.stack([np.array(item) for item in orig_test])
orig_test_tensor = torch.tensor(orig_test_array)
results_test_array = np.stack([np.array(item) for item in results_test])
results_test_tensor = torch.tensor(results_test_array)
targets_test_array = np.stack([np.array(item) for item in y_test])
targets_test_tensor = torch.tensor(targets_test_array)
print("results length after " + str(targets_test_tensor.shape))

train_dataset = customDataset(orig_train_tensor, results_train_tensor, targets_train_tensor, keys_train)
val_dataset = customDataset(orig_val_tensor, results_val_tensor, targets_val_tensor, keys_val)
test_dataset = customDataset(orig_test_tensor, results_test_tensor, targets_test_tensor, keys_test)

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle=True) 
val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle=True) 
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle=True) 

class SaccadeShiftNN(nn.Module):
    def __init__(self):
        super(SaccadeShiftNN, self).__init__()
        self.combine_fc = nn.Linear(4*12*17+3, 128)
        self.combine_fc2 = nn.Linear(128, 64)
        self.combine_fc3 = nn.Linear(64, 32)

        self.output_fc1 = nn.Linear(32, 64)
        self.output_fc2 = nn.Linear(64, 128)
        self.output_fc3 = nn.Linear(128, 4*12*17)

    def forward(self, latent_input, saccade_input):
        latent_flat = latent_input.view(latent_input.size(0), -1)
        saccade_flat = saccade_input.view(saccade_input.size(0), -1)

        combined = torch.cat((latent_flat, saccade_flat), dim=1)
        combined_out = F.relu(self.combine_fc(combined))
        combined_out = F.relu(self.combine_fc2(combined_out))
        combined_out = F.relu(self.combine_fc3(combined_out))

        output = F.relu(self.output_fc1(combined_out))
        output = self.output_fc2(output)
        output = self.output_fc3(output)
        output = output.view(-1, 4, 12, 17)

        return output
        

model = SaccadeShiftNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 800

loss_his = []
loss_val = []

all_outputs = {}
test_latents = {}

for i in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_orig, batch_results, batch_targets, batch_keys in train_dataloader:
        batch_orig = batch_orig.to(device)
        batch_results = batch_results.to(batch_orig.dtype)
        batch_results = batch_results.to(device)
        batch_targets = batch_targets.to(device)

        saccade_shifted_latent = model(batch_orig, batch_results)

        loss = criterion(saccade_shifted_latent, batch_targets)
        # all_outputs.extend(saccade_shifted_latent.cpu().detach().numpy().tolist())
        saccade_shifted_latent = saccade_shifted_latent.detach().cpu().numpy()

        for k, latent in zip(batch_keys, saccade_shifted_latent):
            all_outputs[k] = latent.tolist()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    avg_epoch_loss = train_loss / len(train_dataloader)
    loss_his.append(avg_epoch_loss)
    
    print(f'Epoch [{i+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_orig, batch_results, batch_targets, _ in val_dataloader:
            batch_orig = batch_orig.to(device)
            batch_results = batch_results.to(device)
            batch_targets = batch_targets.to(device)

            saccade_shifted_latent = model(batch_orig, batch_results)
            loss = criterion(saccade_shifted_latent, batch_targets)
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
        test_loss += loss.item()

        saccade_shifted_latent = saccade_shifted_latent.detach().cpu().numpy()

        for k, latent in zip(batch_keys, saccade_shifted_latent):
            test_latents[k] = latent.tolist()

avg_test_loss = test_loss / len(test_dataloader)
print(f'Test Loss: {avg_test_loss:.4f}')

with open('/scratch/gpfs/qw3971/cnn-vae-old/rotated/test2.json', 'w') as json_file:
    json.dump(test_latents, json_file)

plt.figure(figsize=(10, 6))
plt.plot(loss_val)
plt.title('Training Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Save the figure to a file
plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/rotated/test.png', dpi=300, bbox_inches='tight')

print('done') 