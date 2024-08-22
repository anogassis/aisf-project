import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Any, Iterable, Optional, Union, Tuple, List, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from tms.utils.config import DEVICE
from tms.data.dataset import SyntheticBinaryValued
from tms.models.autoencoder import ToyAutoencoder
from tms.utils.utils import generate_init_param, generate_optimal_solution


def create_and_train(
    m: int,
    n: int,
    num_samples: int,
    batch_size: Optional[int] = 1,
    num_epochs: int = 100,
    sparsity: Union[float, int] = 1,
    lr: float = 0.001,
    log_ivl: Iterable[int] = [],
    device=DEVICE,
    momentum=0.9,
    weight_decay=0.0,
    init_kgon: int = None,
    no_bias: bool = False,
    init_zerobias: bool = False,
    prior_std: float = 10.,
    seed: int = 0,
    use_optimal_solution: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Create and train a model using the given parameters.

    Parameters
    ----------
    m : int
        The number of input features.
    n : int
        The number of output features.
    num_samples : int
        The number of training samples.
    batch_size : Optional[int], optional
        The batch size for training, by default 1.
    num_epochs : int, optional
        The number of training epochs, by default 100.
    sparsity : Union[float, int], optional
        The sparsity level of the training data, by default 1.
    lr : float, optional
        The learning rate for the optimizer, by default 0.001.
    log_ivl : Iterable[int], optional
        The intervals at which to log the training progress, by default [].
    device : _type_, optional
        The device to use for training, by default DEVICE.
    momentum : float, optional
        The momentum factor for the optimizer, by default 0.9.
    weight_decay : float, optional
        The weight decay factor for the optimizer, by default 0.0.
    init_kgon : _type_, optional
        The initialization method for the model weights, by default None.
    no_bias : bool, optional
        Whether to exclude bias terms in the model, by default False.
    init_zerobias : bool, optional
        Whether to initialize bias terms to zero, by default False.
    prior_std : _type_, optional
        The standard deviation of the prior distribution for weight initialization, by default 10.
    seed : int, optional
        The random seed for reproducibility, by default 0.
    use_optimal_solution : bool, optional
        Whether to use an optimal solution for weight initialization, by default False.

    Returns
    -------
    logs : pandas.DataFrame
        A DataFrame containing the training logs.
    weights : list
        A list of dictionaries containing the model weights at different training steps.
    """
    torch.manual_seed(seed)

    model = ToyAutoencoder(m, n, final_bias=True)
    init_weights = generate_init_param(n, m, init_kgon, prior_std=prior_std, no_bias=no_bias, init_zerobias=init_zerobias, seed=seed)

    model.embedding.weight.data = torch.from_numpy(init_weights["W"]).float()
    if use_optimal_solution:
        init_weights = generate_optimal_solution(m, n, rot=0.0)

    if "b" in init_weights:
        model.unembedding.bias.data = torch.from_numpy(init_weights["b"].flatten()).float()

    dataset = SyntheticBinaryValued(num_samples, m, sparsity)
    batch_size = batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    logs = pd.DataFrame([{"loss": None, "acc": None, "step": step} for step in log_ivl])

    model.to(device)
    weights = []

    def log(step):
        loss = 0.0
        acc = 0.0
        length = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
                outputs = model(batch)
                loss += criterion(outputs, batch).item() * len(batch) # adding "* len(batch)"
                acc += (outputs.round() == batch).float().sum().item()
                length += len(batch)

        loss /= length
        acc /= length

        logs.loc[logs["step"] == step, ["loss", "acc"]] = [loss, acc]
        weights.append({k: v.cpu().detach().clone().numpy() for k, v in model.state_dict().items()})

    step = 0
    log(step)

    for epoch in tqdm(range(num_epochs), desc="Training"):
        for batch in dataloader:
            batch = batch.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            step += 1

            if step in log_ivl:
                log(step)

    return logs, weights
