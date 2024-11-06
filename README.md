# Evolutionary Reinforcement Learning for Interpretable Decision Making for the Supply Chain
In the context of Industry 4.0, Supply Chain Management (SCM) faces challenges in adopting advanced optimization techniques due to the "black-box" nature of AI-based solvers, which causes reluctance among company stakeholders. This work employs an Interpretable Artificial Intelligence (IAI) approach that combines evolutionary computation with reinforcement learning to generate interpretable decision-making policies in the form of decision trees (DT). This interpretable solver is embedded within a simulation-based optimization framework specifically designed to handle the inherent uncertainties and stochastic behaviors of modern supply chains. To our knowledge, this marks the first attempt to combine IAI solvers with simulation-based optimization for SCM decision-making. The methodology is tested on two supply chain optimization problems, one fictional and one from the real world, and its performance is compared against widely used algorithms. The results reveal that the interpretable approach delivers competitive, and sometimes better, performance, challenging the prevailing notion that there must be a trade-off between interpretability and optimization efficiency. Additionally, the developed framework demonstrates strong potential for industrial applications, offering seamless integration with various Python-based solvers.

<p align="center">
<img src="methodologySchema.png" width="500">
</p>

## Requirements
Before getting started, make sure you have installed all the requirements.
```
pip install -r requirements.txt
```

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

## Contributions
Authors:
- Stefano Genetti, MSc Student University of Trento (Italy), stefano.genetti@studenti.unitn.it
- Giovanni Iacca, Associate Professor University of Trento (Italy), giovanni.iacca@unitn.it

For every type of doubt/question about the repository please do not hesitate to contact us.
