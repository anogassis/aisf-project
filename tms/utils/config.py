import os
import torch
from tms.utils.utils import generate_sparsity_values

DEVICE = os.environ.get(
    "DEVICE",
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu",
)

training_dicts = {
   "debug":
{
    "m": [6],
    "n": [2],
    "num_samples": [100], 
    "batch_size": [1024],
    "num_epochs": [4500],
    "sparsity": generate_sparsity_values(5, 10),
    "lr": [0.005],
    "momentum": [0.9],
    "weight_decay": [0.0],
    "init_kgon": [6],
    "no_bias": [False],
    "init_zerobias": [False],
    "prior_std": [0],
    "seed": [i for i in range(10)],
    "use_optimal_solution": [True],
},
    "1.3.0": 
    {
    "m": [6],
    "n": [2],
    "num_samples": [100], #Later in iteration 2 we will try 1000 samples
    "batch_size": [1024],
    "num_epochs": [20000],
    "sparsity": generate_sparsity_values(5, 10),
    "lr": [0.005],
    "momentum": [0.9],
    "weight_decay": [0.0],
    "init_kgon": [6],
    "no_bias": [False],
    "init_zerobias": [False],
    "prior_std": [0],
    "seed": [i for i in range(50)],
    "use_optimal_solution": [True],
},
    "1.4.0": 
    {
    "m": [6],
    "n": [2],
    "num_samples": [10000],
    "batch_size": [1024],
    "num_epochs": [20000],
    "sparsity": generate_sparsity_values(5, 10),
    "lr": [0.005],
    "momentum": [0.9],
    "weight_decay": [0.0],
    "init_kgon": [6],
    "no_bias": [False],
    "init_zerobias": [False],
    "prior_std": [0],
    "seed": [i for i in range(50)],
    "use_optimal_solution": [True],
},
    "1.5.0":  # When I ran this version, I used the wrong initilisation for the k-gon. Because that function was so far not included in the dictionary. 1.6.0 is the same dictionary (unless I add the k-gon initialisation function)
    {
    "m": [6],
    "n": [2],
    "num_samples": [100], #Later in iteration 2 we will try 1000 samples
    "batch_size": [1024],
    "num_epochs": [20000],
    "sparsity": generate_sparsity_values(5, 10),
    "lr": [0.005],
    "momentum": [0.9],
    "weight_decay": [0.0],
    "init_kgon": [6],
    "no_bias": [False],
    "init_zerobias": [False],
    "prior_std": [0],
    "seed": [i for i in range(50)],
    "use_optimal_solution": [True],
}, 
    "1.6.0": 
    {
    "m": [6],
    "n": [2],
    "num_samples": [100], #Later in iteration 2 we will try 1000 samples
    "batch_size": [1024],
    "num_epochs": [20000],
    "sparsity": generate_sparsity_values(5, 10),
    "lr": [0.005],
    "momentum": [0.9],
    "weight_decay": [0.0],
    "init_kgon": [6],
    "no_bias": [False],
    "init_zerobias": [False],
    "prior_std": [0],
    "seed": [i for i in range(50)],
    "use_optimal_solution": [True],
},

    "1.7.0": 
    #Big run initialized at optimal solution to see if we get interesting things out of the learning coefficients.
    {
    "m": [6],
    "n": [2],
    "num_samples": [1000], 
    "batch_size": [1024],
    "num_epochs": [20000],
    "sparsity": generate_sparsity_values(5, 10),
    "lr": [0.005],
    "momentum": [0.9],
    "weight_decay": [0.0],
    "init_kgon": [6],
    "no_bias": [False],
    "init_zerobias": [False],
    "prior_std": [0],
    "seed": [i for i in range(50)],
    "use_optimal_solution": [True],
},
     "1.8.0":  
    # New quick run to see if the learning coefficients are more interpretable, if we are initializing at a more random spot.
    {
    "m": [6],
    "n": [2],
    "num_samples": [100], 
    "batch_size": [1024],
    "num_epochs": [20000],
    "sparsity": [x for x in generate_sparsity_values(5, 10) if x != 0],
    "lr": [0.005],
    "momentum": [0.9],
    "weight_decay": [0.0],
    "init_kgon": [4],
    "no_bias": [False],
    "init_zerobias": [False],
    "prior_std": [1.],
    "seed": [i for i in range(50)],
    "use_optimal_solution": [False],
},
    }

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
