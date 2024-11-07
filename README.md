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
As regards the **make-or-buy** decision problem the RS baseline given a list of N orders, the algorithm generates a random binary vector **x** of length N. This binary vector is then evaluated using the AnyLogic simulation, which calculates the resulting total revenue based on the sequence of decisions encoded in **x**. The process is repeated for `n_iterations` iterations. At each step, the algorithm tracks and updates the best-performing solution based on the revenue obtained. On the other hand, for the **HFS** task the RS baseline generates and evaluates random permutations of the input jobs over `n_iterations`, continuously tracking the best solution identified during the process.
#### greedy heuristic (GREEDY)
For the **HFS** problem we include in our experimental evaluationa a simple deterministic greedy heuristic, which prioritizes jobs based on their due dates. Specifically, jobs with the earliest due dates are given the highest priority.
#### optuna (OPTUNA)
[Optuna](https://optuna.org/) is a widely used [open-source](https://github.com/optuna/optuna.git) optimizer that employs state-of-the-art algorithms for solving optimization problems. It consists of two main components: a sampler and a pruner. The sampler generates design variable configurations, known as trials, aiming to explore the most promising regions of the solution space. By default, the Tree-structured Parzen Estimator is used. The pruner’s role is to automatically terminate unpromising trials, avoiding exploration of less promising areas of the fitness landscape. Optuna’s default pruning method, the Median Pruner, stops a trial if its best intermediate result is worse than the median of previous trials at the same step. For the **make-or-buy** decision problem, **OPTUNA** generates a binary vector **x** of length N orders during each trial, and the algorithm is executed for `n_trials` trials. In the **HFS** task, **OPTUNA** is executed for `n_trials` trials, where each iteration assigns a priority value between 0 and N to each job in the input, with N representing the total number of laser cutting machine orders to be scheduled.
#### genetic algorithm (GA)
We developed **GA** using the structure provided by the [open-source inspyred Python library](https://github.com/aarongarrett/inspyred.git). For the **make-or-buy** task, in our **GA** implementation, each individual candidate solution, denoted as **x**, is represented by a fixed-length binary genotype, where the length corresponds to the total number of orders. Each gene in **x** encodes a decision for a specific order: either "make" (0) or "buy" (1). The fitness of an individual is determined by the revenue _R_ obtained through the AnyLogic simulation, which evaluates the sequence of make-or-buy decisions encoded in the genotype. Initially, the population is composed by `ga_psize` randomly generated individuals. Following the default setting of inspyred, our **GA** employs rank selection, one-point crossover, bit-flip mutation. Rank selection indicates that individuals are ranked based on their fitness and higher-ranked individuals have a higher chance of being selected for reproduction. One-point crossover is a popular crossover technique. Given two parent individuals, a single crossover point is randomly chosen along the length of the chromosome, and the offspring inherit the genetic material from one parent up to that point and from the other parent after it. The operator is applied with a probability denoted by `ga_cp`. Bit-flip mutation is a mutation operator where the chromosomes are randomly flipped from 0 to 1 or from 1 to 0. Each bit is mutated with probability `ga_mp`. In addition, evolution follows the generational replacement with elitism approach, where the top `ga_elities` individuals, meaning those with the highest fitness, are preserved across generations. The remaining population is replaced with new offspring generated through selection, crossover, and mutation, ensuring the introduction of novel genetic material in each generation and guiding the algorithm towards convergence. The evolutionary process is controlled by a computational budget defined as a maximum number of generations, `ga_gen`.
In line with current trends in HFS research, we include a **GA** implementation also for the **HFS** problem. The population consists of `ga_psize` individuals, where each candidate solution is a list of input jobs arranged in descending priority order. The fitness of each individual is determined by the makespan achieved at the end of the AnyLogic simulation based on the proposed job sequence. The evolutionary algorithm runs for `ga_gen` generations. Rank selection is used to choose parent individuals for crossover, meaning individuals are selected based on their rank in the population, with higher-ranked individuals having a greater chance of being selected. Generational replacement with elitism is employed at each generation, where the entire population is replaced by new offspring, except for the best `ga_elites` individuals, who are preserved to ensure high-quality solutions carry over to the next generation. Mutation and crossover are applied with probabilities `ga_mp` and `ga_cp`, respectively. Since individuals represent a permutation of the input jobs, both evolutionary operators must avoid introducing duplicates in the genotype of the newly generated offspring. The mutation operator randomly swaps genes in an individual a specified number of times, `ga_mn`. For crossover, we use the order crossover (OX) method [originally proposed by Davis](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4f1330b9b790d0adfe80c82c8b2ff27e6cdbf715). The operation of the OX operator can be illustrated using an example from Zbigniew Michalewicz’s book ["Genetic Algorithms + Data Structures = Evolution Programs"](https://books.google.com/books?hl=it&lr=&id=JmyrCAAAQBAJ&oi=fnd&pg=PA1&dq=Genetic+algorithms%2B+data+structures%3D+evolution+programs&ots=YrJPBU9kvs&sig=sfQTKyBiFZggleopODtDV2Zd9qw). Given two parent solutions, `p₁` and `p₂`, with two randomly selected cut points marked by '|':
`p₁` = (1 2 3 | 4 5 6 7 | 8 9)  
`p₂` = (4 5 2 | 1 8 7 6 | 9 3)  

First, the segments between the cut points are copied directly into the offspring:

`o₁` = (x x x | 4 5 6 7 | x x)  
`o₂` = (x x x | 1 8 7 6 | x x)  

Next, starting from the second cut point of one parent, the genes from the other parent are copied in the same order, skipping any genes already present in the offspring. Once the end of the string is reached, the procedure continues from the first place of the list. The final offspring are:

`o₁` = (2 1 8 | 4 5 6 7 | 9 3)  
`o₂` = (3 4 5 | 1 8 7 6 | 9 2)  

This method effectively transfers information about the relative order of jobs from parents to offspring, emphasizing the importance of job sequencing in the resulting solutions.
#### ant colony optimization (ACO)
Ant colony optimization is particularly well-suited for tackling discrete combinatorial optimization problems. In particular, our formulation of the **make-or-buy** decision problem closely resembles the 0-1 knapsack problem. Ant colony optimization has demonstrated effectiveness in solving the knapsack problem in various research efforts. To develop our optimization algorithm **ACO**, we adopt the implementation structure provided by the inspyred Python library. In the inspyred framework, a swarm of `aco_psize` artificial ants constructs potential solutions by iteratively selecting components from a set of trial components **C**. Each trial component **c** ∈ **C** is associated with a desirability score **d(c)**, which is computed as a combination of the pheromone level **τ** and a problem-dependent heuristic value **η**. In our experiments, we found that incorporating problem-specific heuristic information is not necessary for achieving good results, so we set **η=1** for all trial components. The set of trial components **C** consists of 2 ∗ **#orders** elements, represented as **C** = $\{o_1, o_2, . . ., o_\text{orders}, -o_1, -o_2, . . ., -o_\text{orders}\}$. Selecting oi corresponds to outsourcing order i, while selecting -oi means handling order i internally. During the solution construction process, an ant can choose, following a probabilistic transition rule, only from components oi such that neither oi nor -oi has already been included in the partial solution. The fitness of the solution path p, composed of the selected trial components, is determined by the total revenue R generated from the AnyLogic simulation. This fitness value guides the pheromone update process of the trial components. The evolutionary process continues for a maximum number of generations, denoted by `aco_gen`.
Ant colony optimization is well-suited for solving discrete optimization problems, where variables take on discrete values, such as integers. The authors of [[paper](https://link.springer.com/article/10.1007/s00170-007-1048-2)] were the first to apply ant colony optimization to solve the **HFS** problem. Their experiments used benchmarks with a small number of jobs, suggesting that limiting the number of input jobs is necessary to avoid excessive execution times. In our study, we implement **ACO** using the inspyred Python library. Within this implementation, a set $C$ of trial components have to be defined. Each trial component is characterized by a heuristic value $\eta$, a pheromone level \tau, and a desirability $d$. Initially, we defined the discrete set of trial components as $C = (0, j_0),(0, j_1),...,(n−1, j_{n−1})$, where j_0, j_1, . . ., j_{n−1} represent the jobs, and $n$ is the total number of jobs. In this configuration, an ant would construct a solution by selecting components, where choosing component $(i, j_k)$ means that job $j_k$ is placed in the $i$-th position of the schedule. At each step, an ant can only select components that lead to jobs not yet included in the schedule, ensuring that each job appears only once. The fitness of an ant’s path, representing a schedule, is evaluated by the makespan obtained from an AnyLogic simulation following the proposed job order. This approach resulted in excessive execution times for problem instances with a number of jobs $n$ an order of magnitude larger than those considered in [[paper](https://link.springer.com/article/10.1007/s00170-007-1048-2)]. In order to overcome this issue, we revised the ant colony optimization implementation by reducing the set of trial components, thereby decreasing the computational complexity. In our modified approach (**ACO**), the set of trial components is simplified to $C = j_0, . . ., j_n$. Each ant constructs a job schedule by sequentially selecting jobs that have not yet been included in the schedule. The limitation of this method is that the position of jobs in the final schedule is not directly considered when updating pheromone levels. To mitigate this, we modified the pheromone update rule to account for the position of each job in the schedule. Only the pheromone levels of the trial components in the best solution are updated. Since in this case all components in $C$ are part of the best solution (which is a permutation of $C$), the pheromone level $\tau_{j_i}$ for component $j_i$ is updated as follows:
$$\tau_{ij}(t+1) = (1 − \alpha) \cdot \tau_{ij}(t) + \alpha \cdot f(p) \cdot (n − pos(j_i,p))$$
where $p$ is the best solution path (i.e., the schedule of jobs resulting in the smallest makespan), $|p| = |C| = n$, and $\text{pos}(j_i, p)$ is the position of job $j_i$ in the schedule $p$. This approach gives higher weight to jobs placed earlier in the schedule. By default, the ant colony optimization implementation in the inspyred Python library is designed for maximization problems. Since we aim to minimize the makespan, the fitness of a solution constructed by an ant, denoted as $f(p)$, is defined as the reciprocal of the makespan obtained from the AnyLogic simulation based on the proposed schedule. This can be expressed as: $f(p) = \frac{1}{\text{makespan}}$. In this way, lower makespan values will result in higher fitness scores, aligning the problem with the library’s maximization framework. The heuristic value $\eta_{j_i}$, associated with each trail component $j_i$ is defined as \frac{1}{\text{delivery}(j_i)}, where $\text{delivery}(j_i)$ is the due date of job $j_i$. The optimal job configuration is searched by a population of `aco_psize` ants that evolve over `aco_gen` generations.
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
