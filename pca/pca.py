import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

with open('/scratch/gpfs/qw3971/cnn-vae-old/latent_16/latent.json', 'r') as json_file:
    latent = json.load(json_file)

with open('/scratch/gpfs/qw3971/cnn-vae-old/pca/left.json', 'r') as json_file:
    black = json.load(json_file)

with open('/scratch/gpfs/qw3971/cnn-vae-old/pca/right.json', 'r') as json_file:
    white = json.load(json_file)

black_latent = []
white_latent = []

# Retrieve the latent vectors for each image with three elements
for image_filename in black:
    # image_filename = "shifted_" + image_filename
    latent_vector = latent.get(image_filename)
    if latent_vector is not None:
        black_latent.append(latent_vector)

# Retrieve the latent vectors for each image with three elements
for image_filename in white:
    # image_filename = "shifted_" + image_filename
    latent_vector = latent.get(image_filename)
    if latent_vector is not None:
        white_latent.append(latent_vector)

print(len(black_latent))
print(len(white_latent))

# randomly generate random data for two groups (You will feed in your latent vectors for different images)
group1 = torch.tensor(black_latent)  # Group 1 with mean [2, 2, 2]
group2 = torch.tensor(white_latent)  # Group 2 with mean [-2, -2, -2]

group1_flat = group1.view(len(black_latent), -1)
group2_flat = group2.view(len(white_latent), -1)

print("mine 1:", group1_flat.dim())
print("mine 1:", group1_flat.shape)

'''
group1 = torch.randn(50, 3) + torch.tensor([2, 2, 2])
print("theirs 1", group1.dim())
print("mine 1:", group1.shape)
group1 = torch.randn(50, 3) + torch.tensor([-2, -2, -2])
'''

data = torch.cat([group1_flat, group2_flat], dim=0)

# Perform PCA
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data.numpy())
# Separate PCA results for each group  (for you this will be the number of categories. For example same shape/same material has the same color, or similar position has the same color etc)
pca_result_group1 = pca_result[:len(black_latent), :]
pca_result_group2 = pca_result[len(black_latent):len(black_latent)+len(white_latent), :]
# Visualize in 3D PCA space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_result_group1[:, 0], pca_result_group1[:, 1], pca_result_group1[:, 2], label='Left', c='blue')
ax.scatter(pca_result_group2[:, 0], pca_result_group2[:, 1], pca_result_group2[:, 2], label='Right', c='red')
ax.set_title('3D PCA Components Visualization Left/Right')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.savefig('/scratch/gpfs/qw3971/cnn-vae-old/latent_16/pca_left_right.png')

plt.legend()
plt.show()

import plotly.graph_objs as go
import plotly.offline as py

# Assuming pca_result_group1 and pca_result_group2 are your PCA results as in your original code

trace1 = go.Scatter3d(
    x=pca_result_group1[:, 0],
    y=pca_result_group1[:, 1],
    z=pca_result_group1[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        color='blue',                # set color to an array/list of desired values
        opacity=0.8
    )
)

trace2 = go.Scatter3d(
    x=pca_result_group2[:, 0],
    y=pca_result_group2[:, 1],
    z=pca_result_group2[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        color='red',                # set color to an array/list of desired values
        opacity=0.8
    )
)

data = [trace1, trace2]
layout = go.Layout(
    title='3D PCA Components Visualization Three/Seven',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='pca16_left_right.html')
