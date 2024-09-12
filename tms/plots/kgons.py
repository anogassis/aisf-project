import numpy as np
import torch

from src.models.autoencoder import count_kgons
from scipy.spatial import ConvexHull
from typing import Any, Dict, List
from matplotlib import pyplot as plt


def plot_percentage_of_kgons_over_time(weights: Dict[int, List[List[Dict[str, torch.Tensor]]]], steps: List[int], k_values: List[int] = [5, 6], xscales: str = "log", yscales: str = "linear", plot: bool = True, title: str = None) -> None:
    """
    Plot the percentage of k-gons over time for different sparsities.

    Parameters
    ----------
    weights : Dict[int, List[List[Dict[str, torch.Tensor]]]]
        A dictionary containing the weights for different sparsities and time steps.
    steps : List[int]
        A list of time steps.
    k_values : List[int], optional
        A list of k-values representing the k-gons of interest. Default is [5, 6].
    xscales : str, optional
        The scale of the x-axis. Default is "log".
    yscales : str, optional
        The scale of the y-axis. Default is "linear".
    plot : bool, optional
        Whether to plot the results. Default is True.
    title : str, optional
        The title of the plot. If not provided, a default title will be generated.

    Returns
    -------
    None
    """
    plt.figure(figsize=(15, 6))
    
    # Generate a color map to represent different sparsities with a color gradient
    sparsities = sorted(weights.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sparsities)))
    
    # Create a dictionary to store percentages of k-gons for each sparsity
    kgon_percentages = {sparsity: {k: [] for k in k_values} for sparsity in sparsities}
    
    # Iterate over each sparsity
    for sparsity, runs_weights in weights.items():
        # Get the color for the current sparsity
        color = colors[sparsities.index(sparsity)]
        
        # Iterate over each time step
        for step_weights in zip(*runs_weights):  # This transposes the list of lists
            edge_counts = count_kgons(step_weights)
            total_counts = sum(edge_counts.values())
            
            # Calculate percentages for interested k-gons
            for k in k_values:
                percentage = (edge_counts.get(k, 0) / total_counts) * 100
                kgon_percentages[sparsity][k].append(percentage)
        
        # Plot the percentage of k-gons over time for each k-value
        percentages = np.zeros(len(steps))
        for k in k_values:
            percentages += kgon_percentages[sparsity][k][:-1]  # fixme: reduced length by one but not sure why this is longer?
        label = f'Sparsity: {sparsity}, {" ".join([str(k) for k in k_values])}-gons'
        plt.plot(steps, percentages, label=label, color=color)

    plt.xlabel('Step')
    plt.ylabel('Percentage of k-gons')
    plt.xscale(xscales)
    plt.yscale(yscales)
    if not title:
        plt.title(f'Percentage of {", ".join([str(k) for k in k_values])}-gons over Training Steps for Different Sparsities')
    else: 
        plt.title(title)
    plt.legend()

def plot_rate_of_change_of_kgons(weights: Dict[int, List[List[Dict[str, torch.Tensor]]]], steps: List[int], k_values: List[int] = [5, 6], xscale: str = 'log') -> None:
    """
    Plot the rate of change of the percentage of k-gons over training steps for different sparsities.

    Parameters
    ----------
    weights : Dict[int, List[List[Dict[str, torch.Tensor]]]]
        A dictionary containing the weights for different sparsities and training steps.
    steps : List[int]
        A list of training steps.
    k_values : List[int], optional
        A list of k-values representing the number of sides of the polygons, by default [5, 6].
    xscale : str, optional
        The scale of the x-axis, by default 'log'.

    Returns
    -------
    None
    """
    plt.figure(figsize=(15, 6))
    
    sparsities = sorted(weights.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sparsities)))
    
    kgon_percentages = {sparsity: {k: [] for k in k_values} for sparsity in sparsities}
    kgon_rate_of_change = {sparsity: {k: [] for k in k_values} for sparsity in sparsities}
        
    rate_of_change = np.zeros(len(steps) - 1)
    
    for sparsity, runs_weights in weights.items():
        color = colors[sparsities.index(sparsity)]
        
        for step_weights in zip(*runs_weights):
            edge_counts = count_kgons(step_weights)
            total_counts = sum(edge_counts.values())
            
            for k in k_values:
                percentage = (edge_counts.get(k, 0) / total_counts) * 100
                kgon_percentages[sparsity][k].append(percentage)
        
        for k in k_values:
            kgon_rate_of_change[sparsity][k] = np.diff(kgon_percentages[sparsity][k])
        
        for k in k_values:
            rate_of_change += kgon_rate_of_change[sparsity][k]
        label = f'Sparsity: {sparsity}, {" ".join([str(k) for k in k_values])}-gons Rate of Change'
    plt.plot(steps[1:], rate_of_change, label=label, color=color)
    
    plt.xlabel('Step')
    plt.ylabel('Rate of Change of Percentage of k-gons')
    plt.xscale(xscale)
    plt.title(f'Rate of Change of {", ".join([str(k) for k in k_values])}-gons over Training Steps for Different Sparsities')
    plt.legend()
    plt.show()
       
def plot_polygon(
    W: torch.Tensor,
    b=None,
    ax=None,
    ax_bias=None,
    ax_wnorm=None,
    hull_alpha=0.3,
    dW=None,
    dW_scale=0.3,
    orderb=True,
    color="b",
):
    """
    Plot a polygon based on the given vectors. Credits: Edmund Lau.

    Parameters
    ----------
    W : torch.Tensor
        The input tensor containing the vectors.
    b : torch.Tensor, optional
        The bias tensor, by default None.
    ax : matplotlib.axes.Axes, optional
        The axes object to plot on, by default None.
    ax_bias : matplotlib.axes.Axes, optional
        The axes object to plot the bias on, by default None.
    ax_wnorm : matplotlib.axes.Axes, optional
        The axes object to plot the vector norms on, by default None.
    hull_alpha : float, optional
        The alpha value for the convex hull plot, by default 0.3.
    dW : torch.Tensor, optional
        The tensor containing the derivative of W, by default None.
    dW_scale : float, optional
        The scaling factor for the derivative of W, by default 0.3.
    orderb : bool, optional
        Whether to order the bias values, by default True.
    color : str, optional
        The color of the vectors, by default "b".

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

    Raises
    ------
    ValueError
        If W does not have either 2 or 3 rows.
    """
    
    if ax is None:
        if W.shape[0] == 2:
            fig, ax = plt.subplots(1, 1)
        elif W.shape[0] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

    if W.shape[0] == 2:  # 2D case
        # Compute the norms of the columns
        norms = np.linalg.norm(W, axis=0)

        # Normalize a copy of the vectors for angle calculations
        W_normalized = W / norms

        # Compute angles from the x-axis for each vector
        angles = np.arctan2(W_normalized[1, :], W_normalized[0, :])

        # Sort the columns of W by angles
        order = np.argsort(angles)
        W_sorted = W[:, order]

        # Plot the origin
        ax.scatter(0, 0, color="red")

        # Plot the vectors
        for i in range(W_sorted.shape[1]):
            ax.quiver(
                0,
                0,
                W_sorted[0, i],
                W_sorted[1, i],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.003,
            )
        if dW is not None:
            dW = -dW_scale * dW / np.max(np.linalg.norm(dW, axis=0))
            for col in range(W.shape[1]):
                ax.quiver(
                    W[0, col],
                    W[1, col],
                    dW[0, col],
                    dW[1, col],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="r",
                    width=0.005,
                )

        # Connect the vectors to form a polygon
        polygon = np.column_stack((W_sorted, W_sorted[:, 0]))
        ax.plot(polygon[0, :], polygon[1, :], alpha=0.5)

        # Plot the convex hull
        hull = ConvexHull(W.T)
        vs = list(hull.vertices) + [hull.vertices[0]]
        ax.plot(W[0, vs], W[1, vs], "r--", alpha=hull_alpha)

        # Set the aspect ratio of the plot to equal to ensure that angles are displayed correctly
        ax.set_aspect("equal", adjustable="box")

    elif W.shape[0] == 3:  # 3D case
        # Plot the origin
        ax.scatter([0], [0], [0], color="red")

        # Plot the vectors
        for i in range(W.shape[1]):
            ax.plot([0, W[0, i]], [0, W[1, i]], [0, W[2, i]], color)

        # Plot the convex hull
        hull = ConvexHull(W.T)
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(W[0, s], W[1, s], W[2, s], "r--", alpha=hull_alpha)
    else:
        raise ValueError("W must have either 2 or 3 rows")

    if b is not None and ax_bias is not None and W.shape[0]==2:
        
        b_plot = np.ravel(b)
        if orderb:
            b_plot = b_plot[order]
        bar_colors = ["r" if val < 0 else "g" for val in b_plot]
        yticks = np.array(range(1, len(b_plot) + 1))
        ax_bias.barh(
            yticks - 0.4,
            np.abs(b_plot),
            height=0.4,
            color=bar_colors,
            align="edge",
        )
        ax_bias.set_yticks(yticks)
        ax_bias.yaxis.tick_right()
        ax_bias.tick_params(axis="y", labelsize="x-small")
        ax_bias.tick_params(axis="x", labelsize="x-small")

    if ax_wnorm is not None and W.shape[0]==2:
        yticks = np.array(range(1, W.shape[1] + 1))
        wnorms = np.linalg.norm(W, axis=0)
        if orderb:
            wnorms = wnorms[order]
        ax_wnorm.barh(yticks, width=wnorms, height=0.4, color="black", alpha=0.9, align="edge")
    return ax

def plot_polygons(Ws, biases, axes=None, ax_biases=None):
    """
    Plot polygons based on the given weights and biases.

    Parameters
    ----------
    Ws : list
        List of weight matrices for each polygon.
    biases : list
        List of bias vectors for each polygon.
    axes : ndarray, optional
        Array of subplot axes to plot the polygons on. If not provided, a new figure with subplots will be created.
    ax_biases : ndarray, optional
        Array of subplot axes to plot the bias vectors on. If not provided, a new figure with subplots will be created.

    Returns
    -------
    None
    """
    if axes is None:
        fig, axes = plt.subplots(1, len(Ws), figsize=(15, 4))
    if ax_biases is None:
        fig, ax_biases = plt.subplots(1, len(Ws), figsize=(15, 4))

    for ax, W, ax_b, b in zip(axes, Ws, ax_biases, biases):
        plot_polygon(W, b=b, ax=ax, ax_bias=ax_b, ax_wnorm=ax_b)

def plot_losses_and_polygons(steps, losses, highlights, Ws, biases, xscale="log", yscale="log", batch_size=None, run=None, version=None):
    """
    Plot the losses and weight snapshots of polygons.

    Parameters
    ----------
    steps : list
        List of steps.
    losses : list
        List of losses.
    highlights : list
        List of steps to highlight.
    Ws : list
        List of weight snapshots.
    biases : list
        List of biases.
    xscale : str, optional
        Scale of the x-axis. Default is "log".
    yscale : str, optional
        Scale of the y-axis. Default is "log".
    batch_size : int, optional
        Batch size. Default is None.
    run : int, optional
        Run number. Default is None.
    version : int, optional
        Version number. Default is None.
    """
    fig = plt.figure(figsize=(15, 6))

    gs = fig.add_gridspec(3, len(Ws))
    ax_losses = fig.add_subplot(gs[2, :])
    ax_polygons = []
    ax_biases = []

    max_x, min_x = max([np.max(W[0]) for W in Ws]), min([np.min(W[0]) for W in Ws])
    max_y, min_y = max([np.max(W[1]) for W in Ws]), min([np.min(W[1]) for W in Ws])

    for i in range(len(Ws)):
        ax = fig.add_subplot(gs[0, i], adjustable='box') 
        ax.set_aspect('equal')
        ax_polygons.append(ax)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y+0.5)
    for i in range(len(Ws)):
        ax = fig.add_subplot(gs[1, i], adjustable='box')
        ax_biases.append(ax)
        ax.set_xlim(0, 1.5)
        
    ax_losses.plot(steps, losses)
    ax_losses.set_xlabel("Step")
    ax_losses.set_ylabel("Loss")
    ax_losses.set_xscale(xscale)
    ax_losses.set_yscale(yscale)

    for i, step in enumerate(highlights):
        ax_losses.axvline(step, color="gray", linestyle="--")

    plot_polygons(Ws, biases, ax_polygons, ax_biases=ax_biases)
    version_str = f"Version: {version}" if version is not None else ""
    batch_size_str = f"Batch size: {batch_size}" if batch_size is not None else ""
    run_str = f"Run: {run}" if run is not None else ""
    plt.suptitle("Loss and Weight snapshots, " + batch_size_str + " " + run_str + " " + version_str)
    plt.tight_layout()

def plot_experiments(
    results: List[Dict[str, Any]],
    show:bool = True,
    save: bool = False,
    file_name: str = None
    ) -> None:
    """
    Plots the results of the experiments using plot_losses_and_polygons.

    Parameters
    ----------
    results : List[dict]
        A list of dictionaries, each containing the run_id, parameters used, logs, and weights.
    """
    for result in results:
        run_id = result['run_id']
        params = result['parameters']
        logs = result['logs']
        weights = result['weights']

        # Extract steps and losses from logs
        steps = list(logs['step'].values)
        losses = list(logs['loss'].values)

        # Generate highlight steps based on the number of epochs
        num_epochs = params.get('num_epochs', 100)
        num_observations = 50
        plot_steps = [min(steps, key=lambda s: abs(s-i)) for i in [0, 200, 2000, 10000, num_epochs - 1]]
        plot_indices = [steps.index(s) for s in plot_steps]

        # Extract weights at the highlight steps
        Ws = [weights[i]['embedding.weight'] for i in plot_indices]

        # Plot losses and polygons
        plt.figure()
        plot_losses_and_polygons(steps, losses, plot_steps, Ws)

        # Title the plot based on parameters
        keys_in_title = ['run_id','m', 'n' ,'num_samples', 'batch_size', 'sparsity', 'lr']
        title = ', '.join(f'{key}: {value}' for key, value in params.items() if key in keys_in_title)
        plt.suptitle(f'Run ID: {run_id}\n {title}')
        plt.tight_layout()

        if show:
            plt.show()

        if save:
            plt.savefig(f'{file_name}_{run_id}.png')

        plt.close()
