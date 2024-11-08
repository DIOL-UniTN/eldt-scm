from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import random
import numpy as np
import operator
import json
import sys
import inspect
import time
import os
import csv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

def save_tree(nodes: List[int], edges: List[Tuple[int, int]], labels: Dict, filename: str)->None:
    data = {
        "nodes": nodes,
        "edges": edges,
        "labels": labels
    }
    with open(filename, 'w') as f:
        json.dump(data, f)

def read_arguments():
    parser = argparse.ArgumentParser(description="ABCD production optimizer.")
    
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    
    parser.add_argument("--out_dir", type=str, help="Output folder.")
    parser.add_argument('--no_runs', type=int, default=1, help='Number of runs.')

    parser.add_argument('--population_size', type=int, default=100, help='population size for GP.')
    parser.add_argument('--max_generations', type=int, default=50, help='number of generations for GP.')
    parser.add_argument('--mutation_pb', type=float, default=0.2, help='mutation probability for GP.')
    parser.add_argument('--crossover_pb', type=float, default=0.5, help='crossover probability for GP.')
    parser.add_argument('--trnmt_size', type=int, default=3, help='tournament size for GP.')
    parser.add_argument('--hof_size', type=int, default=1, help='size of the hall-of-fame for GP.')

    parser.add_argument("--output_tree", default="tree.json", type=str, help="File path where to store the GP tree.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])
    
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
        plt.savefig(f"{args['out_dir']}/gp.pdf")
        plt.savefig(f"{args['out_dir']}/gp.png")
    else:
        # create directory for saving results
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_path = f"{args['out_dir']}/{current_datetime}"
        os.makedirs(output_folder_path)

        # execution time csv file
        csv_execution_time_file_path = f"{output_folder_path}/exec_gp.csv"
        csv_execution_time_file = open(csv_execution_time_file_path, mode='w', newline='')
        csv_execution_time_writer = csv.writer(csv_execution_time_file)
        csv_execution_time_writer.writerow(["run", "time"])
        csv_execution_time_file.close()

        abcd_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )

        # init design variable setup
        design_variable_setup = abcd_model.get_jvm().abcdproduction.DesignVariableSetup()
        
        def simulation(x, reset=True):
            for orderId, outsourceFlag in enumerate(x):
                if outsourceFlag==1:
                    design_variable_setup.setToOne(orderId)
                else:
                    design_variable_setup.setToZero(orderId)
            
            # pass input setup and run model until end
            abcd_model.setup_and_run(design_variable_setup)
            
            # extract model output or simulation result
            model_output = abcd_model.get_model_output()
            if reset:
                # reset simulation model to be ready for next iteration
                abcd_model.reset()
            
            return model_output.getTotalRevenue()
        
        # read orders from excel file
        df = pd.read_excel("orders.xlsx")
        orders = []
        for _, row in df.iterrows():
            order = {
                'order_id': int(row['id']),
                'components': np.array([int(row['a']), int(row['b']), int(row['c'])]),
                'deadline': pd.to_datetime(row['deadline']).timestamp()
            }
            orders.append(order)

        # input:
        # --> number of components of type A required for the current order
        # --> number of components of type B required for the current order
        # --> number of components of type C required for the current order

        # per risolvere problema dei bool in lt e gt
        # https://stackoverflow.com/a/54365884
        # https://groups.google.com/g/deap-users/c/ggNVMNjMenI?pli=1
        # https://github.com/DEAP/deap/issues/237
        class Bool(object): pass
        pset = gp.PrimitiveSetTyped("MAIN", [int, int, int], str)

        # function set
        def if_then_else(condition, out1, out2):
            return out1 if condition else out2
        def integer(num):
            return num
        #pset.addPrimitive(operator.and_, [Bool, Bool], Bool)
        #pset.addPrimitive(operator.or_, [Bool, Bool], Bool)
        #pset.addPrimitive(operator.not_, [Bool], Bool)
        pset.addPrimitive(operator.lt, [int, int], Bool)
        pset.addPrimitive(operator.gt, [int, int], Bool)
        pset.addPrimitive(if_then_else, [Bool, str, str], str)
        pset.addPrimitive(integer, [int], int)

        # terminal set
        pset.addEphemeralConstant("number", lambda: random.randint(1,20), int)
        pset.addTerminal("outsource", str)
        pset.addTerminal("not_outsource", str)
        pset.addTerminal(False, Bool)
        pset.addTerminal(True, Bool)

        # rename input arguments
        pset.renameArguments(ARG0='a')
        pset.renameArguments(ARG1='b')
        pset.renameArguments(ARG2='c')

        # creator object
        creator.create("FitnessRevenue", base.Fitness, weights=(1.0,))
        creator.create("IndividualRevenue", gp.PrimitiveTree, fitness=creator.FitnessRevenue)

        # toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.IndividualRevenue, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # evalutation function
        def gpEvaluation(individual, orders: Dict[str, Dict]):
            #print(f"individual={individual}")
            # transform the tree expression in a callable function
            gpFunction = toolbox.compile(expr=individual)

            # for each order in order apply the gpFunction for outsourcing decision
            x:List[int] = []
            for order in orders:
                d = gpFunction(order["components"][0],
                               order["components"][1],
                               order["components"][2])
                if d=="not_outsource":
                    x.append(0)
                else:
                    x.append(1)
            #print(f"x: {x}")
            #return random.randint(1,100),
            return simulation(x),
        
        toolbox.register("evaluate", gpEvaluation, orders=orders)
        toolbox.register("select", tools.selTournament, tournsize=args["trnmt_size"])
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        #toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        #toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        # init random generator for reproducible outcomes
        random.seed(args["random_seed"])
        np.random.seed(args["random_seed"])

        for r in range(args["no_runs"]):
            # create directory for saving results of the run
            output_folder_run_path = output_folder_path+"/"+str(r+1)
            os.makedirs(output_folder_run_path)

            # write a txt log file with algorithm configuration and other execution details
            log_file = open(f"{output_folder_run_path}/log.txt", 'w')
            log_file.write(f"algorithm: genetic programming\n")
            log_file.write(f"current date and time: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_file.write(f"no_runs: {args['no_runs']}\n")
            log_file.write(f"population_size: {args['population_size']}\n")
            log_file.write(f"max_generations: {args['max_generations']}\n")
            log_file.write(f"mutation_pb: {args['mutation_pb']}\n")
            log_file.write(f"crossover_pb: {args['crossover_pb']}\n")
            log_file.write(f"trnmt_size: {args['trnmt_size']}\n")
            log_file.write(f"hof_size: {args['hof_size']}\n")
            log_file.write(f"output_tree: {args['output_tree']}\n")
            log_file.write(f"\n===============\n")

            # population size and hall of fame size
            pop = toolbox.population(n=args["population_size"])
            hof = tools.HallOfFame(args["hof_size"])

            # statistics
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            mstats.register("avg", np.mean)
            mstats.register("std", np.std)
            mstats.register("min", np.min)
            mstats.register("max", np.max)

            # run optimization
            start_time = time.time()
            final_pop,logbook=algorithms.eaSimple(  pop,
                                                    toolbox,
                                                    args["crossover_pb"],
                                                    args["mutation_pb"],
                                                    args["max_generations"],
                                                    stats=mstats,
                                                    halloffame=hof,
                                                    verbose=True)
            execution_time = time.time()-start_time
            # store execution time of the run
            csv_execution_time_file = open(csv_execution_time_file_path, mode='a', newline='')
            csv_execution_time_writer = csv.writer(csv_execution_time_file)
            csv_execution_time_writer.writerow([r, execution_time])
            csv_execution_time_file.close()
            
            # plot GP tree
            nodes, edges, labels = gp.graph(hof[0])
            #print(f"nodes: {nodes}")
            #print(f"edges: {edges}")
            #print(f"labels: {labels}")
            save_tree(nodes=nodes, edges=edges, labels=labels, filename=f"{output_folder_run_path}/{args['output_tree']}")

            # plot fitness trends
            # TODO: vedere come si aggiunge chapter statistic per la dimensione dell'albero
            plt_generation = logbook.chapters["fitness"].select("gen")
            plt_fit_min = logbook.chapters["fitness"].select("min")
            plt_fit_max = logbook.chapters["fitness"].select("max")
            plt_fit_avg = logbook.chapters["fitness"].select("avg")
            plt.figure(figsize=(10, 6))
            plt.plot(plt_generation, plt_fit_avg, marker='o', linestyle='-', linewidth=1, markersize=4, label='Average Fitness')
            plt.plot(plt_generation, plt_fit_max, marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
            plt.plot(plt_generation, plt_fit_min, marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Revenue')
            plt.title('Fitness Trend')
            plt.legend()
            plt.grid(True)
            #plt.show()
            plt.savefig(f"{output_folder_run_path}/gp.pdf")
            plt.savefig(f"{output_folder_run_path}/gp.png")

            # best individual
            print("Best individual GP is %s, %s" % (hof[0], hof[0].fitness.values))
            log_file.write("Best individual GP is %s, %s\n" % (hof[0], hof[0].fitness.values))

            # store result csv
            df = pd.DataFrame()
            df["generation"] = plt_generation
            df["eval"] = df["generation"] * args["population_size"]
            df["average_fitness"] = plt_fit_avg
            df["best_fitness"] = plt_fit_max
            df["worst_fitness"] = plt_fit_min
            df.to_csv(f"{output_folder_run_path}/history_gp.csv", sep=",", index=False)
            log_file.close()
        
        abcd_model.close()