# Evolutionary Reinforcement Learning for Interpretable Decision Making for the Supply Chain
*Conference [Evostar 2025](https://www.evostar.org/2025/)* </br>
In the context of Industry 4.0, Supply Chain Management (SCM) faces challenges in adopting advanced optimization techniques due to the "black-box" nature of AI-based solvers, which causes reluctance among company stakeholders. This work employs an Interpretable Artificial Intelligence (IAI) approach that combines evolutionary computation with reinforcement learning to generate interpretable decision-making policies in the form of decision trees (DT). This interpretable solver is embedded within a simulation-based optimization framework specifically designed to handle the inherent uncertainties and stochastic behaviors of modern supply chains. To our knowledge, this marks the first attempt to combine IAI solvers with simulation-based optimization for SCM decision-making. The methodology is tested on two supply chain optimization problems, one fictional and one from the real world, and its performance is compared against widely used algorithms. The results reveal that the interpretable approach delivers competitive, and sometimes better, performance, challenging the prevailing notion that there must be a trade-off between interpretability and optimization efficiency. Additionally, the developed framework demonstrates strong potential for industrial applications, offering seamless integration with various Python-based solvers.

[![arXiv](https://img.shields.io/badge/arXiv-2404.05621-b31b1b.svg)]()

```bibtex
@article{genetti2024influence,
  title={Influence Maximization in Hypergraphs using Multi-Objective Evolutionary Algorithms},
  author={Genetti, Stefano and Ribaga, Eros and Cunegatti, Elia and Lotito, Quintino Francesco and Iacca, Giovanni},
  journal={arXiv preprint arXiv:2405.10187},
  year={2024}
}
```

<p align="center">
<img src="hypergraph-im-visualization.png" width="500">
</p>

## Requirements
Before getting started, make sure you have installed all the requirements.
```
pip install -r requirements.txt
```

## Structure
The repository is structured as follows:
```
    .
    ├── data                            # Hypergraphs dataset
    ├── ea                              # Files implementing the inspyred functions (evaluator, mutator, ...)
    ├── greedy                          # Implementation of the high-degree baseline
    ├── random                          # Implementation of the random baseline
    ├── hdd                             # Implementation of the HDD baseline
    ├── smart_initialization.py         # Code for generating the initial population as described in the paper
    ├── moea.py                         # Source code HN-MOEA
    ├── main.py                         # Code main file
    └── monte_carlo_max_hop.py          # Propagation models
```

## External libraries and codes
### HyperGraphX Python Library
In this implementation in order to represent and handle hypergraphs we use the library **HGX**.
- GitHub: [https://github.com/HGX-Team/hypergraphx.git](https://github.com/HGX-Team/hypergraphx.git)
- Paper:  [https://github.com/HGX-Team/hypergraphx.git](https://github.com/HGX-Team/hypergraphx.git)
### HCI-TM-algorithm
The code of HCI-1 and HCI-2 baseline algorithms analyzed in the paper have been taken from the GitHub repository made available by the original authors.
- GitHub: [https://github.com/QDragon18/Influence-Maximization-based-on-Threshold-Model-in-Hypergraphs.git](https://github.com/QDragon18/Influence-Maximization-based-on-Threshold-Model-in-Hypergraphs.git)
- Paper:  [https://doi.org/10.1063/5.0178329](https://doi.org/10.1063/5.0178329)

## Contribution
Authors:
- Stefano Genetti, MSc Student University of Trento (Italy), stefano.genetti@studenti.unitn.it
- Giovanni Iacca, Associate Professor University of Trento (Italy), giovanni.iacca@unitn.it

For every type of doubts/questions about the repository please do not hesitate to contact us.
