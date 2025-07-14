#----------------------------------------------------------------------------------
# Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection (MeMAE)
#----------------------------------------------------------------------------------
# Author: Gong et al. 2019
# Implementation: Ramiro Saltos Atiencia
# Date: 2025-07-14
# Version: 1.2
#----------------------------------------------------------------------------------

# Libraries
#----------------------------------------------------------------------------------

import os
import scipy.io as sio

from torch.utils.data import *
from torch import nn
from nn_models import MeMAE
from nn_train_functions import *
from plot_functions import *
from eval_functions import *

# Setting up the device
#device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# %%Importing the data
#----------------------------------------------------------------------------------
dataset_file = '3G12D.mat'
data_path = os.path.join('data', dataset_file)

# Load data
mat_data = sio.loadmat(data_path)
data = mat_data['Data']
labels = mat_data['y'].ravel() == 2

# Data dimensionality
num_obs, in_dim = data.shape

# %%Data Preparation
#----------------------------------------------------------------------------------

# Create a TensorDataset using the data
train_dataset = TensorDataset(torch.tensor(data, dtype=torch.float32),
                              torch.tensor(data, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(data, dtype=torch.float32),
                             torch.tensor(labels, dtype=torch.float32))

# Create a DataLoader for each dataset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%Model Configuration
#----------------------------------------------------------------------------------
latent_dim = 2
layer_sizes = [in_dim, 10, 8, 4, latent_dim]
ae_model = MeMAE(layer_sizes, mem_size=250, shrink_threshold=0.0025)
ae_loss_fn = nn.MSELoss()

# %%Model Training
#----------------------------------------------------------------------------------

# Set the max epochs for pretraining and training
train_epochs = 100

# Register the start time
start_time = time.time()

# Run the training phase
results_ae_pd = train_ae_network(ae_model, ae_loss_fn, train_loader, epochs=train_epochs, device=device)

# Register the end time
end_time = time.time()

print(f"Total training time was {end_time - start_time:.2f} seconds.")
print(f"Threads de OpenMP: {torch.get_num_threads()}")

# %%Evaluate the performance
#----------------------------------------------------------------------------------

out_scores = get_outlier_scores(ae_model, test_loader, device)
eval_metrics = eval_model(out_scores, labels)
print(eval_metrics)

# %%Plot the results
#----------------------------------------------------------------------------------

plot_training_loss(results_ae_pd)
plot_mem_ae(data, ae_model, device)