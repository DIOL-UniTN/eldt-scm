from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import random
import numpy as np
import gymnasium as gym
import re
import datetime
import os
import time
import csv
from alpypeopt import AnyLogicModel
from gymnasium import spaces
from deap import creator
from deap import base
from deap import tools
from functools import partial

# decision tree
class DecisionTree:
    def __init__(self, phenotype, leaf, n_actions, learning_rate, discount_factor, epsilon):
        self.current_reward = 0
        self.last_leaf = None

        self.program = phenotype
        self.leaves = {}
        n_leaves = 0

        while "_leaf" in self.program:
            new_leaf = leaf(n_actions=n_actions,
                            learning_rate=learning_rate,
                            discount_factor=discount_factor,
                            epsilon=epsilon)
            leaf_name = "leaf_{}".format(n_leaves)
            self.leaves[leaf_name] = new_leaf

            self.program = self.program.replace("_leaf", "'{}.get_action()'".format(leaf_name), 1)
            self.program = self.program.replace("_leaf", "{}".format(leaf_name), 1)

            n_leaves += 1
        self.exec_ = compile(self.program, "<string>", "exec", optimize=2)

    def get_action(self, input):
        if len(self.program) == 0:
            return None
        variables = {}  # {"out": None, "leaf": None}
        for idx, i in enumerate(input):
            variables["_in_{}".format(idx)] = i
        variables.update(self.leaves)

        exec(self.exec_, variables)

        current_leaf = self.leaves[variables["leaf"]]
        current_q_value = max(current_leaf.q)
        if self.last_leaf is not None:
            self.last_leaf.update(self.current_reward, current_q_value)
        self.last_leaf = current_leaf 
        
        return current_leaf.get_action()
    
    def set_reward(self, reward):
        self.current_reward = reward

    def new_episode(self):
        self.last_leaf = None

    def __call__(self, x):
        return self.get_action(x)

    def __str__(self):
        return self.program

class Leaf:
    def __init__(self, n_actions, learning_rate, discount_factor, epsilon, low=-1, up=1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.parent = None
        self.iteration = [1] * n_actions
        self.epsilon = epsilon

        self.q = np.random.uniform(low, up, n_actions)
        self.last_action = 0

    def get_action(self):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Get the argmax. If there are equal values, choose randomly between them
            best = [None]
            max_ = -float("inf")
            
            for i, v in enumerate(self.q):
                if v > max_:
                    max_ = v
                    best = [i]
                elif v == max_:
                    best.append(i)

            action = np.random.choice(best)

        self.last_action = action
        self.next_iteration()
        return action

    def update(self, reward, qprime):
        if self.last_action is not None:
            lr = self.learning_rate if not callable(self.learning_rate) else self.learning_rate(self.iteration[self.last_action])
            if lr == "auto":
                lr = 1/self.iteration[self.last_action]
            self.q[self.last_action] += lr*(reward + self.discount_factor * qprime - self.q[self.last_action])
    
    def next_iteration(self):
        self.iteration[self.last_action] += 1

    def __repr__(self):
        return ", ".join(["{:.2f}".format(k) for k in self.q])

    def __str__(self):
        return repr(self)
################################################################################
# reinforcement learning environment
class AdigeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 max_date_basement_arrival: int,
                 max_date_electrical_panel_arrival: int,
                 max_delivery_date: int,
                 num_jobs: int,
                 totalAvailableM: int,
                 totalAvailableE: int,
                 totalAvailableR: int,
                 dataset_file_path: str,
                 max_makespan: int,
                 priority_levels: int):
        # machine type encoding
        self.machine_type_encoding: Dict[str, int] = {  # TO BE UPDATED WHEN THE NUMBER OF MACHINE CATEGORIES CHANGE
            "lt7": 0,
            "lt7_p": 1,
            "lt7_ins": 2,
            "lt7_p_ins": 3,
            "lt8": 4,
            "lt8_p": 5,
            "lt8_ula": 6,
            "lt8_p_ula": 7,
            "lt8_12": 8,
            "lt8_p_12": 9,
            "lt8_12_ula": 10,
            "lt8_p_12_ula": 11,
        }
        '''
        self.machine_type_encoding: Dict[str, int] = {  # TO BE UPDATED WHEN THE NUMBER OF MACHINE CATEGORIES CHANGE
            "lt7": 0,
            "lt7_p": 1,
            "lt7_ins": 2,
            "lt7_p_ins": 3
        }
        '''
        '''
        self.machine_type_encoding: Dict[str, int] = {  # TO BE UPDATED WHEN THE NUMBER OF MACHINE CATEGORIES CHANGE
            "lt8": 0,
            "lt8_p": 1,
            "lt8_ula": 2,
            "lt8_p_ula": 3,
            "lt8_12": 4,
            "lt8_p_12": 5,
            "lt8_12_ula": 6,
            "lt8_p_12_ula": 7,
        }
        '''
        '''
        self.machine_type_encoding: Dict[str, int] = {  # TO BE UPDATED WHEN THE NUMBER OF MACHINE CATEGORIES CHANGE
            "lt7": 0,
            "lt8": 1
        }
        '''

        # observation space
        # 1-D vector [machine_type, date_basement_arrival, date_electrical_panel_arrival, date_delivery, remaining_orders]
        low_bounds = np.array([0, 0, 0, 0, 0])
        high_bounds = np.array([len(self.machine_type_encoding)-1, max_date_basement_arrival, max_date_electrical_panel_arrival, max_delivery_date, num_jobs])
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.int64)

        # action space
        #self.action_space = spaces.Discrete(num_jobs)
        self.action_space = spaces.Discrete(priority_levels)

        # anylogic
        self.adige_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )
        self.adige_setup = self.adige_model.get_jvm().adige.AdigeSetup()

        self.design_variable = []
        self.current_job_index = 0
        self.total_num_jobs = num_jobs
        self.priority_levels = priority_levels
        self.totalAvailableM = totalAvailableM
        self.totalAvailableE = totalAvailableE
        self.totalAvailableR = totalAvailableR
        self.dataset_file_path = dataset_file_path
        self.max_makespan = max_makespan
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # read dataset jobs
        input_df = pd.read_csv(self.dataset_file_path)                                                                  # read dataframe
        self.input_jobs_ids: List[int] = input_df["order_id"].tolist()                                                  # list of job identifiers
        self.input_jobs: Dict[int, Dict[str, int]] = input_df.set_index("order_id").T.to_dict()                         # dictionary populated with job features

        # randomly shuffle the order of the jobs populating the environment 
        random.shuffle(self.input_jobs_ids)

        self.design_variable = []
        self.current_job_index = 0

        # observation: [machine_type, date_basement_arrival, date_electrical_panel_arrival, date_delivery, remaining_orders]
        current_job_id = self.input_jobs_ids[self.current_job_index]
        current_job_machine_type = self.machine_type_encoding[self.input_jobs[current_job_id]["machine_type"]]
        current_job_date_basement_arrival = self.input_jobs[current_job_id]["date_basement_arrival"]
        current_job_date_electrical_panel_arrival = self.input_jobs[current_job_id]["date_electrical_panel_arrival"]
        current_job_date_delivery = self.input_jobs[current_job_id]["date_delivery"]
        return np.array([int(current_job_machine_type),
                         int(current_job_date_basement_arrival),
                         int(current_job_date_electrical_panel_arrival),
                         int(current_job_date_delivery),
                         len(self.input_jobs_ids[self.current_job_index:])]), {}

    def step(self, action):
        # action is an integer number representing the priority assigned to the given job
        
        # generate a random number withing a proper range according to the priority level
        range_size = self.total_num_jobs // self.priority_levels
        remainder = self.total_num_jobs % self.priority_levels
        start = action * range_size + min(action, remainder)
        end = start + range_size + (1 if action < remainder else 0)
        self.design_variable.append(random.randint(start, end-1))

        # an episode is terminated if the agent has taken a decision for each order
        terminated = len(self.design_variable)==self.total_num_jobs

        # compute reward
        reward = 0
        if terminated:
            # sort input_jobs_ids according to the priorities written in design_variable
            jobs_with_priorities = list(zip(self.input_jobs_ids, self.design_variable))
            sorted_jobs_with_priorities = sorted(jobs_with_priorities, key=lambda x: x[1], reverse=True)
            sorted_job_ids = [job for job, _ in sorted_jobs_with_priorities]

            # run anylogic simulation to compute the reward
            reward = self.max_makespan-self.simulation(sorted_job_ids)

        # update observation space
        self.current_job_index += 1

        # observation: [machine_type, date_basement_arrival, date_electrical_panel_arrival, date_delivery, remaining_orders]
        if terminated:
            return np.array([0,0,0,0,0]), reward, terminated, False, {}
        current_job_id = self.input_jobs_ids[self.current_job_index]
        current_job_machine_type = self.machine_type_encoding[self.input_jobs[current_job_id]["machine_type"]]
        current_job_date_basement_arrival = self.input_jobs[current_job_id]["date_basement_arrival"]
        current_job_date_electrical_panel_arrival = self.input_jobs[current_job_id]["date_electrical_panel_arrival"]
        current_job_date_delivery = self.input_jobs[current_job_id]["date_delivery"]
        observation = np.array([int(current_job_machine_type),
                                int(current_job_date_basement_arrival),
                                int(current_job_date_electrical_panel_arrival),
                                int(current_job_date_delivery),
                                len(self.input_jobs_ids[self.current_job_index:])])
        return observation, reward, terminated, False, {}

    def render(self):
        print(f"[render] current job; {self.current_job_index}")

    def close(self):
        self.adige_model.close()
        #pass
    
    def simulation(self, x: List[int], reset=True):
            """
            x : list of jobs in descending priority order
            """
            # set available resources
            self.adige_setup.setTotalAvailableM(self.totalAvailableM)
            self.adige_setup.setTotalAvailableE(self.totalAvailableE)
            self.adige_setup.setTotalAvailableR(self.totalAvailableR)

            # set dataset file path
            self.adige_setup.setDatasetFilePath(self.dataset_file_path)

            # assign priorities
            for job_position, job_id in enumerate(x):
                self.adige_setup.setPriority(job_id, len(x)-job_position)
            
            # pass input setup and run model until end
            self.adige_model.setup_and_run(self.adige_setup)
            
            # extract model output or simulation result
            model_output = self.adige_model.get_model_output()
            if reset:
                # reset simulation model to be ready for next iteration
                self.adige_model.reset()
            
            return model_output.getMakespan()
################################################################################
# converter from genotype to python program
class GrammaticalEvolutionTranslator:
    def __init__(self, grammar):
        """
        Initializes a new instance of the Grammatical Evolution
        :param grammar: A dictionary containing the rules of the grammar and their production
        """
        self.operators = grammar

    def _find_candidates(self, string):
        return re.findall("<[^> ]+>", string)

    def _find_replacement(self, candidate, gene):
        key = candidate.replace("<", "").replace(">", "")
        value = self.operators[key][gene % len(self.operators[key])]
        return value

    def genotype_to_str(self, genotype):
        """ This method translates a genotype into an executable program (python) """
        string = "<bt>"
        candidates = [None]
        ctr = 0
        _max_trials = 10
        genes_used = 0

        # Generate phenotype starting from the genotype
        # If the individual runs out of genes, it restarts from the beginning, as suggested by Ryan et al 1998
        while len(candidates) > 0 and ctr <= _max_trials:
            if ctr == _max_trials:
                return "", len(genotype)
            for gene in genotype:
                candidates = self._find_candidates(string)
                if len(candidates) > 0:
                    value = self._find_replacement(candidates[0], gene)
                    string = string.replace(candidates[0], value, 1)
                    genes_used += 1
                else:
                    break
            ctr += 1

        string = self._fix_indentation(string)
        return string, genes_used
            
    def _fix_indentation(self, string):
        # If parenthesis are present in the outermost block, remove them
        if string[0] == "{":
            string = string[1:-1]
        
        # Split in lines
        string = string.replace(";", "\n")
        string = string.replace("{", "{\n")
        string = string.replace("}", "}\n")
        lines = string.split("\n")

        fixed_lines = []
        n_tabs = 0

        # Fix lines
        for line in lines:
            if len(line) > 0:
                fixed_lines.append(" " * 4 * n_tabs + line.replace("{", "").replace("}", ""))

                if line[-1] == "{":
                    n_tabs += 1
                while len(line) > 0 and line[-1] == "}":
                    n_tabs -= 1
                    line = line[:-1]
                if n_tabs >= 100:
                    return "None"

        return "\n".join(fixed_lines)
################################################################################
# definition of the fitness evalutation function
def evaluate_fitness(genotype, episodes, n_actions, learning_rate, discount_factor, epsilon, env, num_jobs):
    leaf = Leaf
    # translate a genotype into an executable program (python)
    phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(genotype)

    #print(f"genotype: {genotype}")
    #print(f"phenotype: {phenotype}")

    # http://www.grammatical-evolution.org/papers/gp98/node2.html
    # individuals run out of genes during the mapping process
    # punish them with a suitably harsh fitness value
    if phenotype=="":
        fitness = 0,
        return fitness, {}

    # behaviour tree based on python code
    bt = DecisionTree(phenotype=phenotype,
                      leaf=leaf,
                      n_actions=n_actions,
                      learning_rate=learning_rate,
                      discount_factor=discount_factor,
                      epsilon=epsilon)

    # reinforcement learning
    global_cumulative_rewards = []
    for iteration in range(episodes):
        obs, _ = env.reset(seed=42)
        bt.new_episode()
        cum_rew = 0
        action = 0
        previous = None

        for t in range(num_jobs):
            action = bt(obs)
            previous = obs[:]
            obs, rew, done, _, _ = env.step(action)
            bt.set_reward(rew)
            cum_rew += rew
            if done:
                break

        bt.set_reward(rew)
        bt(obs)
        global_cumulative_rewards.append(cum_rew)
    env.close()

    fitness = np.mean(global_cumulative_rewards),
    return fitness, bt.leaves
################################################################################
# grammatical evolution
class ListWithParents(list):
    def __init__(self, *iterable):
        super(ListWithParents, self).__init__(*iterable)
        self.parents = []
def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    for i, o in enumerate(offspring):
        o.parents = [i] 

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            offspring[i-1].parents.append(i)
            offspring[i].parents.append(i - 1)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring
def eaSimple(population,
             toolbox,
             cxpb,
             mutpb,
             ngen,
             timeout,
             stats=None,
             halloffame=None,
             verbose=__debug__,
             logfile=None,
             var=varAnd):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
    best = None
    best_leaves = None

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = [*toolbox.map(toolbox.evaluate, invalid_ind)]
    leaves = [f[1] for f in fitnesses]
    fitnesses = [f[0] for f in fitnesses]
    for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
        ind.fitness.values = fit
        if logfile is not None and (best is None or best < fit[0]):
            best = fit[0]
            best_leaves = leaves[i]
            with open(logfile, "a") as log_:
                log_.write("[{}] New best at generation 0 with fitness {}\n".format(datetime.datetime.now(), fit))
                log_.write(str(ind) + "\n")
                log_.write("Leaves\n")
                log_.write(str(leaves[i]) + "\n")

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    if ngen:
        for gen in range(1, ngen + 1):

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = var(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [*toolbox.map(toolbox.evaluate, invalid_ind)]
            leaves = [f[1] for f in fitnesses]
            fitnesses = [f[0] for f in fitnesses]

            for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
                ind.fitness.values = fit
                if logfile is not None and (best is None or best < fit[0]):
                    best = fit[0]
                    best_leaves = leaves[i]
                    with open(logfile, "a") as log_:
                        log_.write("[{}] New best at generation {} with fitness {}\n".format(datetime.datetime.now(), gen, fit))
                        log_.write(str(ind) + "\n")
                        log_.write("Leaves\n")
                        log_.write(str(leaves[i]) + "\n")

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            for o in offspring:
                argmin = np.argmin(map(lambda x: population[x].fitness.values[0], o.parents))

                if o.fitness.values[0] > population[o.parents[argmin]].fitness.values[0]:
                    population[o.parents[argmin]] = o

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
    
    if timeout:
        gen=1
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))

            # Vary the pool of individuals
            offspring = var(offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [*toolbox.map(toolbox.evaluate, invalid_ind)]
            leaves = [f[1] for f in fitnesses]
            fitnesses = [f[0] for f in fitnesses]

            for i, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
                ind.fitness.values = fit
                if logfile is not None and (best is None or best < fit[0]):
                    best = fit[0]
                    best_leaves = leaves[i]
                    with open(logfile, "a") as log_:
                        log_.write("[{}] New best at generation {} with fitness {}\n".format(datetime.datetime.now(), gen, fit))
                        log_.write(str(ind) + "\n")
                        log_.write("Leaves\n")
                        log_.write(str(leaves[i]) + "\n")

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            for o in offspring:
                argmin = np.argmin(map(lambda x: population[x].fitness.values[0], o.parents))

                if o.fitness.values[0] > population[o.parents[argmin]].fitness.values[0]:
                    population[o.parents[argmin]] = o

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)
            
            gen+=1

    return population, logbook, best_leaves
def grammatical_evolution(fitness_function,
                          generations,
                          timeout,
                          cx_prob,
                          m_prob,
                          tournament_size,
                          population_size,
                          hall_of_fame_size,
                          rng,
                          initial_len,
                          max_makespan):
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", ListWithParents, typecode='d', fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # attribute generator
    toolbox.register("attr_bool", rng.randint, 0, 40000)

    # structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, initial_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness_function)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=40000, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(hall_of_fame_size)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_makespan = tools.Statistics(lambda ind: max_makespan-ind.fitness.values[0])
    mstats = tools.MultiStatistics(fitness=stats_fit, makespan=stats_makespan)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    
    if generations:
        pop, log, best_leaves = eaSimple(population=pop,
                                        toolbox=toolbox,
                                        cxpb=cx_prob,
                                        mutpb=m_prob,
                                        ngen=generations,
                                        timeout=None,
                                        stats=mstats,
                                        halloffame=hof,
                                        verbose=True,
                                        logfile="log_ge.txt")
    if timeout:
        pop, log, best_leaves = eaSimple(population=pop,
                                        toolbox=toolbox,
                                        cxpb=cx_prob,
                                        mutpb=m_prob,
                                        ngen=None,
                                        timeout=timeout,
                                        stats=mstats,
                                        halloffame=hof,
                                        verbose=True,
                                        logfile="log_ge.txt")
    
    return pop, log, hof, best_leaves
################################################################################
def read_arguments():
    parser = argparse.ArgumentParser(description="Adige hybrid flow shop scheduling optimizer.")
    
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--out_dir", type=str, help="Output folder.")
    parser.add_argument('--no_runs', type=int, default=1, help='Number of runs.')

    parser.add_argument("--m", type=int, default=1, help="Resources of type M.")
    parser.add_argument("--e", type=int, default=1, help="Resources of type E.")
    parser.add_argument("--r", type=int, default=1, help="Resources of type R.")
    parser.add_argument("--num_machine_types", type=int, help="Number of machine types")
    parser.add_argument("--priority_levels", type=int, help="Number of priority levels")
    parser.add_argument("--max_makespan", type=int, help="Max makespan. Used to avoid handling negative rewards.")
    parser.add_argument("--dataset", type=str, default="data/d.csv", help="File path with the dataset.")

    parser.add_argument('--population_size', type=int, default=10, help='population size.')
    parser.add_argument('--max_generations', type=int, help='number of generations.')
    parser.add_argument('--timeout', type=int, help='computational budget.')
    parser.add_argument('--mutation_pb', type=float, default=0.5, help='mutation probability.')
    parser.add_argument('--crossover_pb', type=float, default=0.5, help='crossover probability.')
    parser.add_argument('--trnmt_size', type=int, default=2, help='tournament size.')
    parser.add_argument('--hall_of_fame_size', type=int, default=1, help='size of the hall-of-fame.')
    parser.add_argument('--genotype_len', type=int, default=100, help='genotype length.')
    parser.add_argument("--episodes", default=10, type=int, help="number of episodes that the agent faces in the fitness evaluation phase")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="the learning rate to be used for Q-learning. Default is: 'auto' (1/k)")
    parser.add_argument("--df", default=0.05, type=float, help="the discount factor used for Q-learning")
    parser.add_argument("--eps", default=0.05, type=float, help="epsilon parameter for the epsilon greedy Q-learning")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])
    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])

    if args["input_csv"]:
        csv_file_path = args["input_csv"]
        df = pd.read_csv(csv_file_path)
        plt.figure(figsize=(10, 6))
        plt.plot(df['generation'][1:], df['average_fitness'][1:], marker='o', linestyle='-', linewidth=1, markersize=4, label='Average Fitness')
        plt.plot(df['generation'][1:], df['best_fitness'][1:], marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
        plt.plot(df['generation'][1:], df['worst_fitness'][1:], marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Makespan')
        plt.title('Fitness Trend')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(f"{args['out_dir']}/ge.pdf")
        plt.savefig(f"{args['out_dir']}/ge.png")
    else:
        # create directory for saving results
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_path = f"{args['out_dir']}/{current_datetime}"
        os.makedirs(output_folder_path)

        # execution time csv file
        csv_execution_time_file_path = f"{output_folder_path}/exec_ge.csv"
        csv_execution_time_file = open(csv_execution_time_file_path, mode='w', newline='')
        csv_execution_time_writer = csv.writer(csv_execution_time_file)
        csv_execution_time_writer.writerow(["run", "time"])
        csv_execution_time_file.close()

        # read the input dataset
        input_df = pd.read_csv(args["dataset"])

        # register the environment
        def env_creator(max_date_basement_arrival: int,
                        max_date_electrical_panel_arrival: int,
                        max_delivery_date: int,
                        num_jobs: int,
                        totalAvailableM: int,
                        totalAvailableE: int,
                        totalAvailableR: int,
                        dataset_file_path: str,
                        max_makespan: int,
                        priority_levels: int):
            return AdigeEnv(max_date_basement_arrival=max_date_basement_arrival,
                            max_date_electrical_panel_arrival=max_date_electrical_panel_arrival,
                            max_delivery_date=max_delivery_date,
                            num_jobs=num_jobs,
                            totalAvailableM=totalAvailableM,
                            totalAvailableE=totalAvailableE,
                            totalAvailableR=totalAvailableR,
                            dataset_file_path=dataset_file_path,
                            max_makespan=max_makespan,
                            priority_levels=priority_levels)
        
        gym.envs.registration.register(
            id="AdigeEnv-v0",
            entry_point=partial(    env_creator,
                                    max_date_basement_arrival=input_df["date_basement_arrival"].max(),
                                    max_date_electrical_panel_arrival=input_df["date_electrical_panel_arrival"].max(),
                                    max_delivery_date=input_df["date_delivery"].max(),
                                    num_jobs=input_df.shape[0],
                                    totalAvailableM=args["m"],
                                    totalAvailableE=args["e"],
                                    totalAvailableR=args["r"],
                                    dataset_file_path=args["dataset"],
                                    max_makespan=args["max_makespan"],
                                    priority_levels=args["priority_levels"]),
        )
        env = gym.make("AdigeEnv-v0")

        # setup of the grammar
        # _in_0: machine type as an integer number
        # _in_1: date_basement_arrival
        # _in_2: date_electrical_panel_arrival
        # _in_3: date_delivery
        # _in_4: remaining_orders
        grammar = {
            "bt": ["<if>"],
            "if": ["if <condition>:{<action>}else:{<action>}", "if <condition_eq>:{<action>}else:{<action>}"],
            "condition_eq": ["_in_0==<const_type_0>"],
            "condition": ["_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(1,5)],
            "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
            "comp_op": [" < ", " > "],
            "const_type_0": [str(n) for n in range(args["num_machine_types"])],
            "const_type_1": [str(n) for n in range(input_df["date_basement_arrival"].max())],
            "const_type_2": [str(n) for n in range(input_df["date_electrical_panel_arrival"].max())],
            "const_type_3": [str(n) for n in range(input_df["date_delivery"].max())],
            "const_type_4": [str(n) for n in range(input_df.shape[0])]
        }
        print(grammar)

        # grammatical evolution
        def fitness_function(x):
            return evaluate_fitness(genotype=x,
                                    episodes=args["episodes"],
                                    n_actions=args["priority_levels"],
                                    learning_rate=args["learning_rate"],
                                    discount_factor=args["df"],
                                    epsilon=args["eps"],
                                    env=env,
                                    num_jobs=input_df.shape[0])

        for r in range(args["no_runs"]):
            # create directory for saving results of the run
            output_folder_run_path = output_folder_path+"/"+str(r+1)
            os.makedirs(output_folder_run_path)

            # clear log file
            logfile = open(f"{output_folder_run_path}/log_ge.txt", "w")
            logfile.write("log\n")
            logfile.write(f"algorithm: grammatical evolution\n")
            logfile.write(f"current date and time: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            logfile.write(f"no_runs: {args['no_runs']}\n")
            logfile.write(f"m: {args['m']}\n")
            logfile.write(f"e: {args['e']}\n")
            logfile.write(f"r: {args['r']}\n")
            logfile.write(f"num_machine_types: {args['num_machine_types']}\n")
            logfile.write(f"max_makespan: {args['max_makespan']}\n")
            logfile.write(f"priority_levels: {args['priority_levels']}\n")
            logfile.write(f"dataset: {args['dataset']}\n")
            logfile.write(f"population_size: {args['population_size']}\n")
            logfile.write(f"max_generations: {args['max_generations']}\n")
            logfile.write(f"timeout: {args['timeout']}\n")
            logfile.write(f"mutation_pb: {args['mutation_pb']}\n")
            logfile.write(f"crossover_pb: {args['crossover_pb']}\n")
            logfile.write(f"trnmt_size: {args['trnmt_size']}\n")
            logfile.write(f"hall_of_fame_size: {args['hall_of_fame_size']}\n")
            logfile.write(f"genotype_len: {args['genotype_len']}\n")
            logfile.write(f"episodes: {args['episodes']}\n")
            logfile.write(f"learning_rate: {args['learning_rate']}\n")
            logfile.write(f"df: {args['df']}\n")
            logfile.write(f"eps: {args['eps']}\n")
            logfile.write(f"\n===============\n")
            logfile.close()
            if args["max_generations"]:
                start_time = time.time()
                pop, log, hof, best_leaves = grammatical_evolution(fitness_function=fitness_function,
                                                                generations=args["max_generations"],
                                                                timeout=None,
                                                                cx_prob=args["crossover_pb"],
                                                                m_prob=args["mutation_pb"],
                                                                tournament_size=args["trnmt_size"],
                                                                population_size=args["population_size"],
                                                                hall_of_fame_size=args["hall_of_fame_size"],
                                                                rng=rng,
                                                                initial_len=args["genotype_len"],
                                                                max_makespan=args["max_makespan"])
                execution_time = time.time()-start_time
                # store execution time of the run
                csv_execution_time_file = open(csv_execution_time_file_path, mode='a', newline='')
                csv_execution_time_writer = csv.writer(csv_execution_time_file)
                csv_execution_time_writer.writerow([r, execution_time])
                csv_execution_time_file.close()
            if args["timeout"]:
                pop, log, hof, best_leaves = grammatical_evolution(fitness_function=fitness_function,
                                                                generations=None,
                                                                timeout=args["timeout"],
                                                                cx_prob=args["crossover_pb"],
                                                                m_prob=args["mutation_pb"],
                                                                tournament_size=args["trnmt_size"],
                                                                population_size=args["population_size"],
                                                                hall_of_fame_size=args["hall_of_fame_size"],
                                                                rng=rng,
                                                                initial_len=args["genotype_len"],
                                                                max_makespan=args["max_makespan"])
            
            # retrive best individual
            phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(hof[0])
            phenotype = phenotype.replace('leaf="_leaf"', '')
            # iterate over all possible leaves
            for k in range(50000):
                key = "leaf_{}".format(k)
                if key in best_leaves:
                    v = best_leaves[key].q
                    phenotype = phenotype.replace("out=_leaf", "out={}".format(np.argmax(v)), 1)
                else:
                    break
            print("Best individual GE is %s, %s" % (hof[0], args["max_makespan"]-hof[0].fitness.values[0]))
            print(f"Phenotype: {phenotype}")

            # write best individual on file
            logfile = open(f"{output_folder_run_path}/log_ge.txt", "a")
            logfile.write(str(log) + "\n")
            logfile.write(str(hof[0]) + "\n")
            logfile.write(phenotype + "\n")
            logfile.write("best_fitness: {}".format(hof[0].fitness.values[0]))
            logfile.close()

            # plot fitness trends
            plt_generation = log.chapters["makespan"].select("gen")
            plt_fit_min = log.chapters["makespan"].select("min")
            plt_fit_max = log.chapters["makespan"].select("max")
            plt_fit_avg = log.chapters["makespan"].select("avg")
            #plt.figure(figsize=(10, 6))
            #plt.plot(plt_generation, plt_fit_avg, marker='o', linestyle='-', linewidth=1, markersize=4, label='Average Fitness')
            #plt.plot(plt_generation, plt_fit_max, marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
            #plt.plot(plt_generation, plt_fit_min, marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
            #plt.xlabel('Generation')
            #plt.ylabel('Makespan')
            #plt.title('Fitness Trend')
            #plt.legend()
            #plt.grid(True)
            #plt.show()

            # best individual
            print("Best individual GE is %s, %s" % (hof[0], args["max_makespan"]-hof[0].fitness.values[0]))

            # store result csv
            df = pd.DataFrame()
            df["generation"] = plt_generation
            df["eval"] = df["generation"] * (args["episodes"] * args["population_size"])
            df["average_fitness"] = plt_fit_avg
            df["best_fitness"] = plt_fit_max
            df["worst_fitness"] = plt_fit_min
            df.to_csv(f"{output_folder_run_path}/history_ge.csv", sep=",", index=False)

            env.reset()