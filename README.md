# Evolutionary Reinforcement Learning for Interpretable Decision Making for the Supply Chain
In the context of Industry 4.0, Supply Chain Management (SCM) faces challenges in adopting advanced optimization techniques due to the "black-box" nature of AI-based solvers, which causes reluctance among company stakeholders. This work employs an Interpretable Artificial Intelligence (IAI) approach that combines evolutionary computation with reinforcement learning to generate interpretable decision-making policies in the form of decision trees (DT). This interpretable solver is embedded within a simulation-based optimization framework specifically designed to handle the inherent uncertainties and stochastic behaviors of modern supply chains. To our knowledge, this marks the first attempt to combine IAI solvers with simulation-based optimization for SCM decision-making. The methodology is tested on two supply chain optimization problems, one fictional and one from the real world, and its performance is compared against widely used algorithms. The results reveal that the interpretable approach delivers competitive, and sometimes better, performance, challenging the prevailing notion that there must be a trade-off between interpretability and optimization efficiency. Additionally, the developed framework demonstrates strong potential for industrial applications, offering seamless integration with various Python-based solvers.

<p align="center">
<img src="methodologySchema.png" width="500">
</p>

---

## Requirements
Before getting started, make sure you have installed all the requirements.
```
pip install -r requirements.txt
```

---

## Algorithms
The optimization algorithms we implemented are divided into two categories based on how they handle the input list of customer orders. **Schedule-as-a-whole approaches** consider the entire list of orders collectively. In contrast, **policy-generating approaches** concentrate on training a policy that processes each order individually, making decisions based on the specific features of that order.
### schedule-as-a-whole approaches
#### random search (RS)
#### greedy heuristic (GREEDY)
#### optuna (OPTUNA)
#### genetic algorithm (GA)
#### ant colony optimization (ACO)

### policy-generating approaches
#### reinforcement learning (RL)
#### genetic programming (GP)
#### evolutionary learning decision trees (ELDT)

---

## Hyperparameter setting
To ensure a fair comparison, each algorithm was allocated 5000 AnyLogic simulation executions. Hyperparameters affecting the computational budget were adjusted accordingly, while other parameters were set to their library defaults.

### schedule-as-a-whole approaches

#### RANDOM

| Parameter      | Value |
|----------------|-------|
| `n_iterations` | 5000  |

#### OPTUNA

| Parameter      | Value |
|----------------|-------|
| `n_trials`     | 5000  |

#### GA

| Parameter       | Value |
|-----------------|-------|
| `ga_psize`      | 10    |
| `ga_gen`        | 500   |
| `ga_mp`         | 1     |
| `ga_cp`         | 1     |
| `ga_mn`         | 1     |
| `ga_elites`     | 1     |

#### ACO

| Parameter       | Value |
|-----------------|-------|
| `aco_gen`       | 500   |
| `aco_psize`     | 10    |
| `q_0`           | 0.5   |
| `α` (alpha)     | 0.1   |
| `δ` (delta)     | 1.0   |
| `ρ` (rho)       | 0.1   |

### policy-generating approaches

#### RL

| Parameter       | Value |
|-----------------|-------|
| `rl_eval`       | 5000 |

#### GP

| Parameter       | Value |
|-----------------|-------|
| `gp_gen`        | 10    |
| `gp_psize`      | 10    |
| `gp_mp`         | 10    |
| `gp_cp`         | 10    |
| `gp_tsize`      | 10    |
| `gp_minT`       | 1     |
| `gp_maxT`       | 3     |
| `gp_minM`       | 0     |
| `gp_maxM`       | 2     |

#### ELDT

| Parameter       | Value  |
|-----------------|--------|
| `g_l`           | 100    |
| `g_max`         | 40000  |
| `p_size`        | 5      |
| `c_p`           | 0.5    |
| `m_p`           | 0.5    |
| `ge_gen`        | 200    |
| `t_size`        | 2      |
| `α` (alpha)     | 0.001  |
| `γ` (gamma)     | 0.05   |
| `ε` (epsilon)   | 0.05   |
| `n_episodes`    | 5      |

---

## Contributions
Authors:
- Stefano Genetti, MSc Student University of Trento (Italy), stefano.genetti@studenti.unitn.it
- Giovanni Iacca, Associate Professor University of Trento (Italy), giovanni.iacca@unitn.it

For every type of doubt/question about the repository please do not hesitate to contact us.
