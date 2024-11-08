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
class ABCDEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        # observation space
        low_bounds = np.array([0, 0, 0], dtype=np.int32)
        high_bounds = np.array([20, 20, 20], dtype=np.int32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.int32)

        # action space
        self.action_space = spaces.Discrete(2)

        # anylogic
        self.abcd_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )
        self.design_variable_setup = self.abcd_model.get_jvm().abcdproduction.DesignVariableSetup()   # init design variable setup
        
        self.design_variable = []
        self.current_order_id = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # read the exel file of orders
        df = pd.read_excel("orders.xlsx")

        self.orders = []
        for _, row in df.iterrows():
            order = {
                'order_id': int(row['id']),
                'components': np.array([int(row['a']), int(row['b']), int(row['c'])]),
                'deadline': pd.to_datetime(row['deadline']).timestamp()
            }
            self.orders.append(order)
        
        self.design_variable = []
        self.current_order_id = 0

        return self.flat_orders(self.orders), {}

    def step(self, action):
        # action is 0 or 1
        self.design_variable.append(action)
        
        # an episode is terminated if we have taken a decision for each order
        terminated = len(self.design_variable)==100
        
        # compute reward
        reward = self.simulation(self.design_variable) if terminated else 0
        
        # update observation space
        self.current_order_id += 1
        observation = self.orders[self.current_order_id:]

        return self.flat_orders(observation), reward, terminated, False, {}

    def render(self):
        print(f"[render] current order: {self.current_order_id}")

    def close(self):
        self.abcd_model.close()

    def simulation(self, x, reset=True):
        for orderId, outsourceFlag in enumerate(x):
            if outsourceFlag==1:
                self.design_variable_setup.setToOne(orderId)
            else:
                self.design_variable_setup.setToZero(orderId)
        
        # pass input setup and run model until end
        self.abcd_model.setup_and_run(self.design_variable_setup)
        
        # extract model output or simulation result
        model_output = self.abcd_model.get_model_output()
        if reset:
            # reset simulation model to be ready for next iteration
            self.abcd_model.reset()
        return model_output.getTotalRevenue()

    def flat_orders(self, orders):
        if(len(orders)==0):
            return np.array([0, 0, 0])
        order = orders[0]
        order_id = order["order_id"]
        components_a = order["components"][0]
        components_b = order["components"][1]
        components_c = order["components"][2]
        deadline = order["deadline"]
        return np.array([components_a, components_b, components_c])
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
def evaluate_fitness(genotype, episodes, n_actions, learning_rate, discount_factor, epsilon, env):
    leaf = Leaf
    # translate a genotype into an executable program (python)
    phenotype, _ = GrammaticalEvolutionTranslator(grammar).genotype_to_str(genotype)

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
    #e = gym.make("ABCDEnv-v0")
    for iteration in range(episodes):
        obs, _ = env.reset(seed=42)
        bt.new_episode()
        cum_rew = 0
        action = 0
        previous = None

        for t in range(100):
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

    return population, logbook, best_leaves
def grammatical_evolution(fitness_function,
                          generations,
                          cx_prob,
                          m_prob,
                          tournament_size,
                          population_size,
                          hall_of_fame_size,
                          rng,
                          initial_len):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log, best_leaves = eaSimple(population=pop,
                                     toolbox=toolbox,
                                     cxpb=cx_prob,
                                     mutpb=m_prob,
                                     ngen=generations,
                                     stats=stats,
                                     halloffame=hof,
                                     verbose=True,
                                     logfile="log_ge.txt")
    
    return pop, log, hof, best_leaves
################################################################################
def read_arguments():
    parser = argparse.ArgumentParser(description="ABCD production optimizer.")
    
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    parser.add_argument("--out_dir", type=str, help="Output folder.")
    parser.add_argument('--no_runs', type=int, default=1, help='Number of runs.')

    parser.add_argument('--population_size', type=int, default=10, help='population size.')
    parser.add_argument('--max_generations', type=int, default=50, help='number of generations.')
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
        plt.plot(df['generation'], df['average_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Average Fitness')
        plt.plot(df['generation'], df['best_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
        plt.plot(df['generation'], df['worst_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Revenue')
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

        # register the environment
        def env_creator():
            return ABCDEnv()
        gym.envs.registration.register(
            id='ABCDEnv-v0',
            entry_point=env_creator,
        )
        env = gym.make("ABCDEnv-v0")

        # setup of the grammar
        grammar = {
            "bt": ["<if>"],
            "if": ["if <condition>:{<action>}else:{<action>}"],
            "condition": ["_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(3)],
            "action": ["out=_leaf;leaf=\"_leaf\"", "<if>"],
            "comp_op": [" < ", " > "],
            "const_type_0": [str(n) for n in range(21)],
            "const_type_1": [str(n) for n in range(21)],
            "const_type_2": [str(n) for n in range(21)],
        }
        print(grammar)

        # grammatical evolution
        def fitness_function(x):
            return evaluate_fitness(genotype=x,
                                    episodes=args["episodes"],
                                    n_actions=2,
                                    learning_rate=args["learning_rate"],
                                    discount_factor=args["df"],
                                    epsilon=args["eps"],
                                    env=env)
        
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
            logfile.write(f"population_size: {args['population_size']}\n")
            logfile.write(f"max_generations: {args['max_generations']}\n")
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

            start_time = time.time()
            pop, log, hof, best_leaves = grammatical_evolution(fitness_function=fitness_function,
                                                            generations=args["max_generations"],
                                                            cx_prob=args["crossover_pb"],
                                                            m_prob=args["mutation_pb"],
                                                            tournament_size=args["trnmt_size"],
                                                            population_size=args["population_size"],
                                                            hall_of_fame_size=args["hall_of_fame_size"],
                                                            rng=rng,
                                                            initial_len=args["genotype_len"])
            execution_time = time.time()-start_time
            # store execution time of the run
            csv_execution_time_file = open(csv_execution_time_file_path, mode='a', newline='')
            csv_execution_time_writer = csv.writer(csv_execution_time_file)
            csv_execution_time_writer.writerow([r, execution_time])
            csv_execution_time_file.close()
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
            print("Best individual GP is %s, %s" % (hof[0], hof[0].fitness.values))
            print(f"Phenotype: {phenotype}")

            # write best individual on file
            logfile = open(f"{output_folder_run_path}/log_ge.txt", "a")
            logfile.write(str(log) + "\n")
            logfile.write(str(hof[0]) + "\n")
            logfile.write(phenotype + "\n")
            logfile.write("best_fitness: {}".format(hof[0].fitness.values[0]))
            logfile.close()

            # plot fitness trends
            plt_generation = log.select("gen")
            plt_fit_min = log.select("min")
            plt_fit_max = log.select("max")
            plt_fit_avg = log.select("avg")
            #plt.figure(figsize=(10, 6))
            #plt.plot(plt_generation, plt_fit_avg, marker='o', linestyle='-', linewidth=1, markersize=4, label='Average Fitness')
            #plt.plot(plt_generation, plt_fit_max, marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
            #plt.plot(plt_generation, plt_fit_min, marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
            #plt.xlabel('Generation')
            #plt.ylabel('Revenue')
            #plt.title('Fitness Trend')
            #plt.legend()
            #plt.grid(True)
            #plt.show()

            # best individual
            print("Best individual GE is %s, %s" % (hof[0], hof[0].fitness.values))

            # store result csv
            df = pd.DataFrame()
            df["generation"] = plt_generation
            df["eval"] = df["generation"] * (args["episodes"] * args["population_size"])
            df["average_fitness"] = plt_fit_avg
            df["best_fitness"] = plt_fit_max
            df["worst_fitness"] = plt_fit_min
            df.to_csv(f"{output_folder_run_path}/history_ge.csv", sep=",", index=False)

            env.reset()

        