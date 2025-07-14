import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F

class MemoryModule(nn.Module):
    def __init__(self, mem_size, latent_dim, shrink_threshold):
        super().__init__()
        self.mem_dim = mem_size
        self.feature_dim = latent_dim
        self.shrink_threshold = shrink_threshold
        self.memory = nn.Parameter(torch.empty(mem_size, latent_dim))
        nn.init.xavier_uniform_(self.memory)

    def forward(self, query):
        # Normalize query and memory for cosine similarity
        query_norm = F.normalize(query, p=2, dim=1)
        memory_norm = F.normalize(self.memory, p=2, dim=1)

        # Compute attention weights: (batch_size, feature_dim) @ (feature_dim, mem_dim) -> (batch_size, mem_dim)
        att_weight = torch.matmul(query_norm, memory_norm.t())
        att_weight = F.softmax(att_weight, dim=1)

        if self.shrink_threshold > 0:
            # Apply hard shrinkage. This encourages sparsity in attention weights
            att_weight = ((F.relu(att_weight - self.shrink_threshold) * att_weight)/
                          (torch.abs(att_weight - self.shrink_threshold) + 1e-12))

            # Re-normalize after shrinkage
            att_weight = F.normalize(att_weight, p=1, dim=1)

        # Retrieve memory-augmented encoded vector
        z_hat = torch.matmul(att_weight, self.memory)

        return z_hat, att_weight

# Define an Autoencoder model with Glorot initialization
# Note the AE architecture must be modified for each specific dataset.
class MeMAE(nn.Module):
    def __init__(self, layer_sizes, mem_size=100, shrink_threshold=0.0025):
        """
        A class to construct an autoencoder-like neural network model consisting
        of an encoder-decoder structure. This architecture builds both the encoder
        and decoder based on a configurable list of layer sizes, and uses the
        LeakyReLU activation function between layers, where appropriate.
        Weights of the model are initialized upon instantiation.

        :param layer_sizes: A list that defines the dimensions of each layer in the autoencoder
            structure. The first element is the input size, the last is the latent or bottleneck
            size, and the intermediate elements are the hidden layer sizes. Used to configure
            both the encoder and decoder parts of the autoencoder.
        :type layer_sizes: list[int]
        :param mem_size: The number of memory elements to store in the memory module.
        :type mem_size: int
        :param shrink_threshold: The shrinkage threshold to use in the memory module.
        :type shrink_threshold: float
        """
        super().__init__()

        # Build the encoder
        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                # Add a LeakyReLU activation function between layers except for the last one
                encoder_layers.append(nn.LeakyReLU(0.1))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build the decoder
        decoder_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i - 1]))
            if i > 1:
                # Add a LeakyReLU activation function between layers except for the first one
                decoder_layers.append(nn.LeakyReLU(0.1))
        self.decoder = nn.Sequential(*decoder_layers)

        # Get the latent dimension from the last layer size
        self.latent_dim = layer_sizes[-1]

        # Instantiate the memory module
        self.memory_module = MemoryModule(mem_size=mem_size, latent_dim=self.latent_dim, shrink_threshold=shrink_threshold)

        # Initialize the weights of the model using Glorot initialization
        self._initialize_weights()

    def forward(self, x):
        x = self.encoder(x)
        z_hat, _ = self.memory_module(x)
        x = self.decoder(z_hat)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    fan_in, fan_out = module.weight.size(1), module.weight.size(0)
                    std = float(torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out))).item())
                    with torch.no_grad():
                        module.bias.uniform_(-std, std)