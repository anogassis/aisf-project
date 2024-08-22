import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.nn import functional as F
from typing import Iterable, Optional, Callable, Dict, List, Any
from scipy.spatial import ConvexHull

class ToyAutoencoder(nn.Module):
    """
    Basic Network class for linear transformation with non-linear activations
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_instances: int = 1,
        tied: bool = True,
        final_bias: bool = False,
        hidden_bias: bool = False,
        nonlinearity: Callable = F.relu,
        unit_weights: bool = False,
        standard_magnitude: bool = False,
        initial_scale_factor: float = 1.0,
        initial_bias: Optional[torch.Tensor] = None,
        initial_embed: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Set the dimensions and parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_instances = n_instances
        self.nonlinearity = nonlinearity
        self.tied = tied
        self.final_bias = final_bias
        self.unit_weights = unit_weights
        self.standard_magnitude = standard_magnitude

        # Define the input layer (embedding)
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias=hidden_bias)

        # Set initial embeddings if provided
        if initial_embed is not None:
            self.embedding.weight.data = initial_embed

        # Define the output layer (unembedding)
        self.unembedding = nn.Linear(self.hidden_dim, self.input_dim, bias=final_bias)

        # Set initial bias if provided
        if initial_bias is not None:
            self.unembedding.bias.data = initial_bias

        # If standard magnitude is set, normalize weights and maintain average norm
        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim=0).mean()
            self.embedding.weight.data = (
                F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
            )

        # If unit weights is set, normalize weights
        if self.unit_weights:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0)

        # Tie the weights of embedding and unembedding layers
        if tied:
            self.unembedding.weight = torch.nn.Parameter(self.embedding.weight.transpose(0, 1))


    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network
        """
        # Apply the same steps for weights as done during initialization
        if self.unit_weights:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0)

        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim=0).mean()
            self.embedding.weight.data = (
                F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
            )

        if self.tied:
            self.unembedding.weight.data = self.embedding.weight.data.transpose(0, 1)

        x = self.embedding(x)
        x = self.unembedding(x)
        x = self.nonlinearity(x)

        return x

def calculate_convex_hull_vertices(W: torch.Tensor) -> int:
    """
    Calculate the number of vertices of the convex hull of the points represented by the columns of W.
    
    Parameters
    ----------
    W : torch.Tensor
        A 2xN matrix where each column represents a point in 2D space.
    
    Returns
    -------
    int
        The number of vertices of the convex hull.
    """
    if W.shape[0] != 2:
        raise ValueError("The weight matrix W must have 2 rows.")
    
    # Convert the tensor to a numpy array if it isn't already
    if isinstance(W, torch.Tensor):
        W = W.cpu().detach().numpy()
    
    hull = ConvexHull(W.T)
    return len(hull.vertices)  # The number of vertices is the same as the number of edges

def count_kgons(W: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Counts the number of k-gons in a list of weight matrices.

    Parameters
    ----------
    W : List[Dict[str, Any]]
        A list of weight matrices.

    Returns
    -------
    Dict[int, int]
        A dictionary where the keys represent the number of edges in a k-gon
        and the values represent the count of k-gons with that number of edges.
    """
    edge_counts = {}
    
    # Process each weight matrix
    for full_w in W:
        num_edges = classify_kgon(full_w)
        if num_edges in edge_counts:
            edge_counts[num_edges] += 1
        else:
            edge_counts[num_edges] = 1

    return edge_counts

import numpy as np
from scipy.spatial import ConvexHull
from typing import Any

def classify_5_gon(W: np.ndarray, b: np.ndarray) -> Any:
    """
    Classify a given set of weights and biases as a 5-gon or not.

    Parameters
    ----------
    W : np.ndarray
        The weight matrix of shape (n, m), where n is the number of input features and m is the number of output neurons.
    b : np.ndarray
        The bias vector of shape (m,), where m is the number of output neurons.

    Returns
    -------
    Any
        Returns 5 if the given weights and biases form a 5-gon, "5+" if there are positive biases not part of the 5-gon, and "not a 5-gon" otherwise.
    """
    if W.shape[0] == 2:
        W = W.T

    # Compute the convex hull
    hull = ConvexHull(W)
    
    # Check if the number of vertices is equal to 5
    if len(hull.vertices) != 5:
        return "not a 5-gon"
    
    # Check if any of the non-vertex biases are large negative
    non_vertex_biases = np.delete(b, hull.vertices)

    # Check for any positive bias that is not part of the convex hull vertices
    non_hull_positive_bias = np.any(non_vertex_biases > 0)

    if not non_hull_positive_bias:
        return 5
    elif non_hull_positive_bias:
        return "5+"
    else:
        return 'not a 5-gon'

def classify_kgon(W: Dict[str, torch.Tensor]) -> int:
    """
    Classify the k-gon based on the given weights.

    Parameters
    ----------
    W : Dict[str, torch.Tensor]
        A dictionary containing the weights for the model.

    Returns
    -------
    int
        The number of edges in the k-gon.

    Notes
    -----
    This function calculates the number of edges in the k-gon based on the given weights.
    If the number of edges is 5, it calls the `classify_5_gon` function with the embedding weights and unembedding bias.
    Otherwise, it returns the number of edges.

    """
    embedding_w = W["embedding.weight"]
    edges = calculate_convex_hull_vertices(embedding_w)
    if edges == 5:
        return classify_5_gon(embedding_w, W["unembedding.bias"])
    return edges

def compute_kgon_percentages(weights: Dict[int, List[List[Dict[str, torch.Tensor]]]], steps: List[int], k_values: List[int] = [5, 6]) -> Dict[int, Dict[int, List[float]]]:
    """
    Compute the percentages of k-gons over time for different sparsity levels.

    Parameters
    ----------
    weights : Dict[int, List[List[Dict[str, torch.Tensor]]]]
        A dictionary containing the weights for different sparsity levels and time steps.
        The keys represent the sparsity levels, and the values are lists of weight matrices for each time step.
    steps : List[int]
        A list of time steps.
    k_values : List[int], optional
        A list of k-values representing the number of edges in a k-gon, by default [5, 6].

    Returns
    -------
    Dict[int, Dict[int, List[float]]]
        A dictionary containing the percentages of k-gons over time for each sparsity level and k-value.
        The outer dictionary has sparsity levels as keys, and the inner dictionary has k-values as keys.
        The values are lists of percentages of k-gons at each time step.
    """
    sparsities = sorted(weights.keys())
    
    # Create a dictionary to store percentages of k-gons for each sparsity
    kgon_percentages = {sparsity: {k: [] for k in k_values} for sparsity in sparsities}
    
    # Iterate over each sparsity
    for sparsity, runs_weights in weights.items():
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
    return kgon_percentages