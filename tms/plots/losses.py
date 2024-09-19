import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tms.models.autoencoder import ToyAutoencoder
from tms.data.dataset import SyntheticBinaryValued
from tms.plots.kgons import plot_losses_and_polygons


def plot_results_by_indices(results, indices):
    """
    Plot the results of the experiment.

    Parameters
    ----------
    results : list
        A list of dictionaries containing the experiment results.
    plot_number : int, optional
        The maximum number of plots to display, by default 5.
    """
    
    sparsity = [result['parameters']['sparsity'] for result in results]
    for sparse_value in sparsity:
        print(f"Plot polygons for sparsity={sparse_value}")
        for index in indices:
            STEPS = results[index]['parameters']['log_ivl']
            if results[index]['parameters']['sparsity'] != sparse_value:
                continue
            logs = results[index]['logs']

            losses = [logs.loc[logs['step'] == s, 'loss'].values[0] for s in STEPS]

            NUM_EPOCHS = results[index]['parameters']['num_epochs']
            PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
            PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
            Ws = [results[index]['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
            biases = [results[index]['weights'][i]['unembedding.bias'] for i in PLOT_INDICES]
            model = ToyAutoencoder(6, 2, final_bias=True)
            new_weights = {}
            for idx, ndarray in results[index]['weights'][PLOT_INDICES[-1]].items():
                new_weights[idx] = torch.from_numpy(ndarray)

            criterion = nn.MSELoss()
        
            model.load_state_dict(new_weights)

            test_set = SyntheticBinaryValued(10000, 6, sparse_value)
            mean_loss_test = 0
            for sample in test_set:
                output = model(sample)
                mean_loss_test += criterion(output, sample)
            # print("Mean loss test:")
            print(f"index: {index}")
            print(mean_loss_test/10000)

            plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases)
            plt.show()
            

def plot_results(results, plot_number=5):
    """
    Plot the results of the experiment.

    Parameters
    ----------
    results : list
        A list of dictionaries containing the experiment results.
    plot_number : int, optional
        The maximum number of plots to display, by default 5.
    """
    
    sparsity = [result['parameters']['sparsity'] for result in results]
    for sparse_value in sparsity:
        plotted = 0
        print(f"Plot polygons for sparsity={sparse_value}")
        for index in range(len(results)):
            
            STEPS = results[index]['parameters']['log_ivl']
            if results[index]['parameters']['sparsity'] != sparse_value:
                continue
            else:
                if plotted >= plot_number:
                    continue
                plotted += 1
            logs = results[index]['logs']

            losses = [logs.loc[logs['step'] == s, 'loss'].values[0] for s in STEPS]

            NUM_EPOCHS = results[index]['parameters']['num_epochs']
            PLOT_STEPS = [min(STEPS, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, NUM_EPOCHS - 1]]
            PLOT_INDICES = [STEPS.index(s) for s in PLOT_STEPS]
            Ws = [results[index]['weights'][i]['embedding.weight'] for i in PLOT_INDICES]
            biases = [results[index]['weights'][i]['unembedding.bias'] for i in PLOT_INDICES]
            model = ToyAutoencoder(6, 2, final_bias=True)
            new_weights = {}
            for idx, ndarray in results[index]['weights'][PLOT_INDICES[-1]].items():
                new_weights[idx] = torch.from_numpy(ndarray)

            criterion = nn.MSELoss()
        
            model.load_state_dict(new_weights)

            test_set = SyntheticBinaryValued(10000, 6, sparse_value)
            mean_loss_test = 0
            for sample in test_set:
                output = model(sample)
                mean_loss_test += criterion(output, sample)
            # print("Mean loss test:")
            print(f"index: {index}")
            print(mean_loss_test/10000)

            plot_losses_and_polygons(STEPS, losses, PLOT_STEPS, Ws, biases)
            plt.show()
            