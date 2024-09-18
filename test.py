import os
import torch

import tms
import tms.training
import tms.training.experiments
import tms.training.train

from tms.utils.config import DEVICE

torch.manual_seed(1)

DEVICE = torch.device(DEVICE)
NUM_CORES = int(os.environ.get("NUM_CORES", 1))

test_dict = {
    "m": [6],
    "n": [2],
    "num_samples": [100],
    "batch_size": [300],
    "num_epochs": [2000],
    "sparsity": [1],
    "lr": [0.001],
    "momentum": [0.9],
    "weight_decay": [0.0],
    "init_kgon": [6],
    "no_bias": [False],
    "init_zerobias": [False],
    "prior_std": [10.],
    "seed": [42],
    "use_optimal_solution": [False],
}

results = tms.training.experiments.run_experiments(
    test_dict, 
    tms.training.train.create_and_train,
    save=True,
    file_name="test"
)