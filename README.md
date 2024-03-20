# saccade_VAE_FFN

This is all the code used in training the datasets and analyzing the results of the VAE-saccade project for the Buschman Lab. 

We utilized a Variational Autoencoder (VAE) from https://github.com/LukeDitria/CNN-VAE attached to a vgg19 head to enhance object detection for the model to learn to represent each image as a latent vector of dimensions 12 by 17 by 3. 

Next, we built a simple 2-layer feedforward neural network to take in a latent vector and a saccade vector (in radians of how far the camera rotated) to predict what the rotated image's resulting latent vector should be.

We performed PCA to determine how the VAE is storing information in the latent vector and decoded the predicted latent vectors from the FFN with the trained VAE to analyze how similar the predicted and actual images were.
