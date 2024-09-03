# aisf-project
This repository is for the capstone project of the AI Safety Fundamentals - AI Alignment course by BlueDot Impact.

# Results

We explored how the solutions in the problem from the toy model of superposition change in the low sparsity regime.
We first initialized the models as six-gons (the optimal solution for 6 input-parameters), which puzzelingly lead to 0 correlation between the loss and the llc within models trained on data of the same sparsity. We then ran another run where we initialized the models as 4-gons like in *[Chen et al. Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition](https://arxiv.org/abs/2310.06301)* which on average lead to worse solutions in the non-sparse regime, but the best solutions tended to be better.

![image](https://raw.githubusercontent.com/anogassis/aisf-project/c33865e412ccef176766245292ef89d416da8c6a/results/loss_vs_llc_epoch_13.png)
![image](https://raw.githubusercontent.com/anogassis/aisf-project/main/results/loss_vs_llc_epoch_85.png)
![image](https://raw.githubusercontent.com/anogassis/aisf-project/main/results/loss_vs_llc_epoch_526.png)
![image](https://raw.githubusercontent.com/anogassis/aisf-project/main/results/loss_vs_llc_epoch_3243.png)
![image](https://raw.githubusercontent.com/anogassis/aisf-project/main/results/loss_vs_llc_epoch_20000.png)
![video](https://github.com/anogassis/aisf-project/raw/main/results/polygon_animation_sparsity_0.42624657926256726_index_0.mp4)
![video](https://github.com/anogassis/aisf-project/blob/main/results/polygon_animation_sparsity_0.6708070121920944_index_50.mp4)
![video](https://github.com/anogassis/aisf-project/blob/main/results/polygon_animation_sparsity_0.8111243971624382_index_100.mp4)




- [ ] there was also this thing in the original paper relating to the sparsity. We should probably check how our results relate to that.
