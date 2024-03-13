# PCA analysis of images

We performed PCA analyis of the latent vectors of the images that the VAE produced after training to try and understand how the VAE is storing the data so that the FFN can learn to combine that information with a saccade vector. 

We performed PCA analysis with a variety of criteria, including number of objects in the images, color of the objects (i.e. black or white), and location, and find a great correlation between images of less objects vs images of a lot of objects. 

We also examined the latent vector of a scene with one objects with another that had two objects (one of which is identitical to the one in the first image). We found the difference between the two latent vectors and added it to a third latent vector that corresponded to an empty scene, and indeed found that the resulting latent vector produced an image with just the second object that was placed in addition in the two-objects image versus the one-object image.
