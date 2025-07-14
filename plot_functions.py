import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS

def plot_mem_ae(data, model, device):
    """
    Plots the MeMAE result both latent space and data space based on given inputs.
    If the data dimensionality exceeds two, multidimensional scaling (MDS) is applied for visualization
    in the data space.

    :param data: The input dataset to be processed and visualized
        in the latent space and data space. It should be in
        the form of a numpy array with samples as rows
        and features as columns.
    :param model: The trained model used to map the input dataset
        to the latent space. The model should be callable and
        return latent space embeddings.
    :param device: The device on which transformations (with the
        model) should be executed. Typically 'cpu' or 'cuda'.
    :return: None
    """

    model.eval()
    with torch.no_grad():
        phi_x = model.memory_module(model.encoder(torch.tensor(data, dtype=torch.float32, device=device)))[0].cpu().detach().numpy()

    # Plot the latent space
    if phi_x.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(phi_x[:, 0], phi_x[:, 1], label='Encoded Data')
        plt.legend()
        plt.show()

    # Plot the data space
    if data.shape[1] > 2:
        # Use MDS with the original data
        mds = MDS(n_components=2, random_state=42)
        x_mds = mds.fit_transform(data)

        plt.figure(figsize=(10, 8))
        plt.scatter(x_mds[:, 0], x_mds[:, 1], label='Data Space')
        plt.legend()
        plt.show()


def plot_training_loss(pd_results, fig_size=(10, 6)):
    """
    Visualizes the training and test loss over epochs.

    :param pd_results: A pandas dataframe containing 'epoch', 'train loss' and optionally 'test loss' columns
    :param fig_size: The figure dimensions (width, height). Defaults to (10, 6).
    """

    # Style configuration
    sns.set_style("whitegrid")

    # Figure setup
    plt.figure(figsize=fig_size)

    # Plot training loss
    plt.plot(
        pd_results["epoch"],
        pd_results["train loss"],
        label='Training Loss',
        marker='x'
    )

    # Add val loss line if available
    if "val loss" in pd_results:
        plt.plot(
            pd_results["epoch"],
            pd_results["val loss"],
            label='Validation Loss',
            marker='o'
        )

    # Add a test loss line if available
    if "test loss" in pd_results:
        plt.plot(
            pd_results["epoch"],
            pd_results["test loss"],
            label='Test Loss',
            marker='o'
        )

    # Configure labels and title
    plt.title('Loss Evolution per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_accuracy(pd_results, fig_size=(10, 6)):
    """
    Visualizes the model accuracy over epochs.

    :param pd_results: A pandas dataframe containing 'epoch', 'train Acc' and
        optionally 'test Acc' and 'val Acc' columns
    :param fig_size: The figure dimensions (width, height). Defaults to (10, 6).
    """

    # Style configuration
    sns.set_style("whitegrid")

    # Figure setup
    plt.figure(figsize=fig_size)

    # Plot training loss
    plt.plot(
        pd_results["epoch"],
        pd_results["train Acc"],
        label='Training Accuracy',
        marker='x'
    )

    # Add val loss line if available
    if "val Acc" in pd_results:
        plt.plot(
            pd_results["epoch"],
            pd_results["val Acc"],
            label='Validation Accuracy',
            marker='o'
        )

    # Add a test loss line if available
    if "test Acc" in pd_results:
        plt.plot(
            pd_results["epoch"],
            pd_results["test Acc"],
            label='Test Accuracy',
            marker='o'
        )

    # Configure labels and title
    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()