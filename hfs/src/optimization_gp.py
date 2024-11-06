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

    parser.add_argument("--m", type=int, default=1, help="Resources of type M.")
    parser.add_argument("--e", type=int, default=1, help="Resources of type E.")
    parser.add_argument("--r", type=int, default=1, help="Resources of type R.")
    parser.add_argument("--num_machine_types", type=int, help="Number of machine types")
    parser.add_argument("--priority_levels", type=int, help="Number of priority levels")
    parser.add_argument("--dataset", type=str, default="data/d.csv", help="File path with the dataset.")

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
        plt.ylabel('Makespan')
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

        adige_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )

        # init design variable setup
        adige_setup = adige_model.get_jvm().adige.AdigeSetup()
        
        # read the input dataset
        input_df = pd.read_csv(args["dataset"])
        input_jobs:List[Dict] = input_df.to_dict(orient="records")

        def simulation(x: List[int],
                       totalAvailableM: int,
                       totalAvailableE: int,
                       totalAvailableR: int,
                       datasetFilePath: str,
                       reset=True):
            """
            x : list of jobs in descending priority order
            totalAvailableM : number of resources of type M available
            totalAvailableE : number of resources of type E available
            totalAvailableR : number of resources of type R available
            datasetFilePath : file path of the csv file storing the dataset
            """
            assert len(x)==len(set(x))
            # set available resources
            adige_setup.setTotalAvailableM(totalAvailableM)
            adige_setup.setTotalAvailableE(totalAvailableE)
            adige_setup.setTotalAvailableR(totalAvailableR)

            # set dataset file path
            adige_setup.setDatasetFilePath(datasetFilePath)

            # assign priorities
            for job_position, job_id in enumerate(x):
                adige_setup.setPriority(job_id, len(x)-job_position)
            
            # pass input setup and run model until end
            adige_model.setup_and_run(adige_setup)
            
            # extract model output or simulation result
            model_output = adige_model.get_model_output()
            if reset:
                # reset simulation model to be ready for next iteration
                adige_model.reset()
            
            return model_output.getMakespan()

        # setup of the grammar
        # ARG0: machine type as an integer number
        # ARG1: date_basement_arrival
        # ARG2: date_electrical_panel_arrival
        # ARG3: date_delivery
        # ARG4: remaining_orders
        # to solve bool problem in lt e gt
        # https://stackoverflow.com/a/54365884
        # https://groups.google.com/g/deap-users/c/ggNVMNjMenI?pli=1
        # https://github.com/DEAP/deap/issues/237
        class Bool(object): pass
        class MachineType(object): pass
        class DateBArrival(object): pass
        class DateEArrival(object): pass
        class DateDelivery(object): pass
        class MachineType_input(object): pass
        class DateBArrival_input(object): pass
        class DateEArrival_input(object): pass
        class DateDelivery_input(object): pass
        class PriorityLevel(object): pass
        pset = gp.PrimitiveSetTyped("MAIN",
                                    [   MachineType_input,
                                        DateBArrival_input,
                                        DateEArrival_input,
                                        DateDelivery_input
                                    ],
                                    PriorityLevel)

        # function set
        def if_then_else(condition, out1, out2):
            return out1 if condition else out2
        def machine_type_identity(x):
            return x
        def date_b_identity(x):
            return x
        def date_e_identity(x):
            return x
        def date_d_identity(x):
            return x
        def machine_type_input_identity(x):
            return x
        def date_b_input_identity(x):
            return x
        def date_e_input_identity(x):
            return x
        def date_d_input_identity(x):
            return x
        pset.addPrimitive(operator.eq, [MachineType_input, MachineType], Bool)
        pset.addPrimitive(operator.lt, [DateBArrival_input, DateBArrival], Bool)
        pset.addPrimitive(operator.lt, [DateEArrival_input, DateEArrival], Bool)
        pset.addPrimitive(operator.lt, [DateDelivery_input, DateDelivery], Bool)
        pset.addPrimitive(operator.gt, [DateBArrival_input, DateBArrival], Bool)
        pset.addPrimitive(operator.gt, [DateEArrival_input, DateEArrival], Bool)
        pset.addPrimitive(operator.gt, [DateDelivery_input, DateDelivery], Bool)
        pset.addPrimitive(if_then_else, [Bool, PriorityLevel, PriorityLevel], PriorityLevel)
        pset.addPrimitive(machine_type_identity, [MachineType], MachineType)
        pset.addPrimitive(date_b_identity, [DateBArrival], DateBArrival)
        pset.addPrimitive(date_e_identity, [DateEArrival], DateEArrival)
        pset.addPrimitive(date_d_identity, [DateDelivery], DateDelivery)
        pset.addPrimitive(machine_type_input_identity, [MachineType_input], MachineType_input)
        pset.addPrimitive(date_b_input_identity, [DateBArrival_input], DateBArrival_input)
        pset.addPrimitive(date_e_input_identity, [DateEArrival_input], DateEArrival_input)
        pset.addPrimitive(date_d_input_identity, [DateDelivery_input], DateDelivery_input)

        # terminal set
        #pset.addEphemeralConstant("number", lambda: random.randint(1,20), int)
        for n in range(args["num_machine_types"]):
            pset.addTerminal(n, MachineType)
        for n in range(input_df["date_basement_arrival"].max()):
            pset.addTerminal(n, DateBArrival)
        for n in range(input_df["date_electrical_panel_arrival"].max()):
            pset.addTerminal(n, DateEArrival)
        for n in range(input_df["date_delivery"].max()):
            pset.addTerminal(n, DateDelivery)
        for n in range(args["priority_levels"]):
            pset.addTerminal(n, PriorityLevel)
        pset.addTerminal(False, Bool)
        pset.addTerminal(True, Bool)

        # rename input arguments
        pset.renameArguments(ARG0='mt')
        pset.renameArguments(ARG1='db')
        pset.renameArguments(ARG2='de')
        pset.renameArguments(ARG3='dd')

        # creator object
        creator.create("FitnessMakespan", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMakespan)

        # toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
        #toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        # evalutation function
        def gpEvaluation(   individual,
                            jobs: List[Dict],
                            num_priority_levels: int,
                            totalAvailableM: int,
                            totalAvailableE: int,
                            totalAvailableR: int,
                            datasetFilePath: str,
                            rng: random.Random):
            #print(f"individual={individual}")
            # transform the tree expression in a callable function
            gpFunction = toolbox.compile(expr=individual)

            # shuffle the list of jobs randomly
            jobs = rng.sample(jobs, len(jobs))
            jobs_ids = [job["order_id"] for job in jobs]

            # define priority level range size according to the number of jobs and the number of priority levels
            range_size = len(jobs) // num_priority_levels
            remainder = len(jobs) % num_priority_levels
            
            # for each job apply the gpFunction
            jobs_priority_levels:List[int] = []
            for job in jobs:
                priority_level = gpFunction(job["machine_type"],
                                            job["date_basement_arrival"],
                                            job["date_electrical_panel_arrival"],
                                            job["date_delivery"])
                
                # assign priority to the job generating a random number according to the priority level
                start = priority_level * range_size + min(priority_level, remainder)
                end = start + range_size + (1 if priority_level < remainder else 0)
                jobs_priority_levels.append(random.randint(start, end-1))

            # sort input_jobs_ids according to the priorities written in jobs_priority_levels
            jobs_with_priorities = list(zip(jobs_ids, jobs_priority_levels))
            sorted_jobs_with_priorities = sorted(jobs_with_priorities, key=lambda x: x[1], reverse=True)
            sorted_job_ids = [job for job, _ in sorted_jobs_with_priorities]

            return simulation(  x=sorted_job_ids,
                                totalAvailableM=totalAvailableM,
                                totalAvailableE=totalAvailableE,
                                totalAvailableR=totalAvailableR,
                                datasetFilePath=datasetFilePath),
        
        toolbox.register("evaluate", gpEvaluation,  jobs=input_jobs,
                                                    num_priority_levels=args["priority_levels"],
                                                    totalAvailableM=args["m"],
                                                    totalAvailableE=args["e"],
                                                    totalAvailableR=args["r"],
                                                    datasetFilePath=args["dataset"],
                                                    rng=rng)
        toolbox.register("select", tools.selTournament, tournsize=args["trnmt_size"])
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        for r in range(args["no_runs"]):
            # create directory for saving results of the run
            output_folder_run_path = output_folder_path+"/"+str(r+1)
            os.makedirs(output_folder_run_path)
            
            # write a txt log file with algorithm configuration and other execution details
            log_file = open(f"{output_folder_run_path}/log_gp.txt", 'w')
            log_file.write(f"algorithm: genetic programming\n")
            log_file.write(f"current date and time: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_file.write(f"no_runs: {args['no_runs']}\n")
            log_file.write(f"random_seed: {args['random_seed']}\n")
            log_file.write(f"out_dir: {args['out_dir']}\n")
            log_file.write(f"m: {args['m']}\n")
            log_file.write(f"e: {args['e']}\n")
            log_file.write(f"r: {args['r']}\n")
            log_file.write(f"num_machine_types: {args['num_machine_types']}\n")
            log_file.write(f"priority_levels: {args['priority_levels']}\n")
            log_file.write(f"dataset: {args['dataset']}\n")
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
            save_tree(nodes=nodes, edges=edges, labels=labels, filename=f"{output_folder_run_path}/{args['output_tree']}")

            # plot fitness trends
            # TODO: investigate how to include statistics about tree size
            plt_generation = logbook.chapters["fitness"].select("gen")
            plt_fit_min = logbook.chapters["fitness"].select("min")
            plt_fit_max = logbook.chapters["fitness"].select("max")
            plt_fit_avg = logbook.chapters["fitness"].select("avg")
            plt.figure(figsize=(10, 6))
            plt.plot(plt_generation, plt_fit_avg, marker='o', linestyle='-', linewidth=1, markersize=4, label='Average Fitness')
            plt.plot(plt_generation, plt_fit_max, marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
            plt.plot(plt_generation, plt_fit_min, marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Makespan')
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
        
        adige_model.close()