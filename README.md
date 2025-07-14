# Simplified PyTorch Implementation of Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection

This [PyTorch](https://pytorch.org/) implementation of the *Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection* (MeMAE) is a simplified version based on the paper by Gong et al. (2019) (doi: 10.48550/arXiv.1904.02639). This code provides a streamlined PyTorch implementation that focuses on the core concepts of the paper while maintaining the essential functionality. The memory module was adapted from the [MeMAE GitHub repository](https://github.com/donggong1/memae-anomaly-detection).

## Overview

Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection (MeMAE) is an unsupervised anomaly detection algorithm that enhances traditional autoencoders with a memory module in the latent space to better detect anomalies.

The project includes:
- Implementation of the MeMAE method based on the Gong et al. (2019) paper.
- Modified autoencoder architecture with a memory module.
- Visualization tools for both latent space and data space
- Evaluation metrics for anomaly detection performance

## Algorithm Description

The MeMAE algorithm combines autoencoder reconstruction with a memory module in the latent space:

1. **Training Phase**: The autoencoder is trained with a combined loss function that includes:
   - Reconstruction loss: Measures how well the autoencoder can reconstruct the input data
   - Memory module: Stores prototypical patterns to enhance reconstruction of normal data

2. **Anomaly Detection Phase**: Anomalies are detected based on a combined score that considers:
   - Reconstruction error: How well the input can be reconstructed
   - Memory addressing: How well the input matches stored memory patterns

The objective function for MeMAE combines these components to create a more robust anomaly detection method that leverages the reconstruction capabilities of autoencoders and the pattern-matching of memory modules.

## Requirements
- Python 3.12
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Scikit-learn >= 1.2.0
- tqdm >= 4.65.0
- SciPy >= 1.10.0

All dependencies are listed in `requirements.txt`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rjsaltos1989/MeMAE.git
   cd MeMAE
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n memae python=3.12
   conda activate memae
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `main.py`: Main script to run the MeMAE algorithm
- `nn_models.py`: Neural network model definitions (MeMAE model with memory module)
- `nn_train_functions.py`: Functions for training the autoencoder
- `eval_functions.py`: Functions for evaluating the MeMAE model
- `plot_functions.py`: Functions for visualizing results

## Usage

### Basic Usage

1. Modify the dataset path in `main.py` to point to your data:
   ```python
   dataset_path = '/path/to/your/data/'
   dataset_file = 'YourDataset.mat'
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

### Customization

You can customize the following parameters in `main.py`:

- `latent_dim`: Dimension of the latent space (default: 2)
- `train_epochs`: Number of epochs for MeMAE training (default: 100)
- `batch_size`: Batch size for training (default: 32)
- `mem_size`: Size of the memory module (default: 100)
- `shrink_threshold`: Threshold for memory addressing (default: 0.0025)

### Input Data Format

The code expects data in MATLAB .mat format with:
- 'Data': Matrix where rows are samples and columns are features
- 'y': Vector of labels where anomalies are labeled as 2

## Example Results

When running the code, you'll get:
1. Training loss plots showing the convergence of the model.
2. Visualization of the data in the latent space.
3. Performance metrics including AUC-ROC, AUC-PR, F1-Score, and Recall

## References

- MeMAE Paper: Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, Anton van den Hengel. Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection. arXiv preprint arXiv:1904.02639, 2019.
- Memory Module Adaptation: [donggong1/memae-anomaly-detection](https://github.com/donggong1/memae-anomaly-detection)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
