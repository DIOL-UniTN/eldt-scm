from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import random
import inspyred
import statistics
import time
import csv

class AdigeACS(inspyred.swarm.ACS):
    """Represents an Ant Colony System discrete optimization algorithm.
    
    This class is built upon the ``EvolutionaryComputation`` class making
    use of an external archive. It assumes that candidate solutions are
    composed of instances of ``TrailComponent``.
    
    Public Attributes:
    
    - *components* -- the full set of discrete components for a given problem
    - *initial_pheromone* -- the initial pheromone on a trail (default 0)
    - *evaporation_rate* -- the rate of pheromone evaporation (default 0.1)
    - *learning_rate* -- the learning rate used in pheromone updates 
      (default 0.1)
    
    """
    def __init__(self, random, components):
        inspyred.swarm.ACS.__init__(self, random, components)
    
    def _internal_archiver(self, random, population, archive, args):
        best = max(population)
        if len(archive) == 0:
            archive.append(best)
        else:
            arc_best = max(archive)
            if best > arc_best:
                archive.remove(arc_best)
                archive.append(best)
            else:
                best = arc_best
        for c in self.components:
            c.pheromone = ((1 - self.evaporation_rate) * c.pheromone + 
                           self.evaporation_rate * self.initial_pheromone)
        
        bestCandidateSchedule = [c.element for c in best.candidate]
        for c in self.components:
            if c in best.candidate:
                c_position = bestCandidateSchedule.index(c.element)
                c.pheromone = ((1 - self.learning_rate) * c.pheromone + 
                               self.learning_rate * best.fitness)*(len(bestCandidateSchedule)-c_position)
        return archive

class Job:
    def __init__(self, order_id: int, machine_type: str, date_basement_arrival: int, 
                 date_electrical_panel_arrival: int, date_delivery: int):
        self.order_id = order_id
        self.machine_type = machine_type
        self.date_basement_arrival = date_basement_arrival
        self.date_electrical_panel_arrival = date_electrical_panel_arrival
        self.date_delivery = date_delivery
    
    def __repr__(self):
        return (f"Job(order_id={self.order_id}, machine_type='{self.machine_type}', "
                f"date_basement_arrival={self.date_basement_arrival}, "
                f"date_electrical_panel_arrival={self.date_electrical_panel_arrival}, "
                f"date_delivery={self.date_delivery})")

class ACO_adige(inspyred.benchmarks.Benchmark):
    def __init__(self,
                 jobs:List[Job],
                 totalAvailableM: int,
                 totalAvailableE: int,
                 totalAvailableR: int,
                 datasetFilePath: str,
                 max_makespan: int):
        inspyred.benchmarks.Benchmark.__init__(self, len(jobs))

        # ant colony optimization attributes
        self.components = [inspyred.swarm.TrailComponent((job.order_id), value=1/job.date_delivery) for job in jobs]
        self.bias = 0.5
        self.bounder = inspyred.ec.DiscreteBounder([job.order_id for job in jobs])
        self.maximize = True
        self._use_ants = False

        # available humand resources and dataset file path
        self.totalAvailableM = totalAvailableM
        self.totalAvailableE = totalAvailableE
        self.totalAvailableR = totalAvailableR
        self.datasetFilePath = datasetFilePath
        self.max_makespan = max_makespan
    
    def constructor(self, random, args):
        # return a candidate solution for an ant colony optimization
        self._use_ants = True
        candidate = []
        while len(candidate) < len(self.components):
            # find feasible components
            feasible_components = []
            if len(candidate)==0:
                feasible_components = self.components
            else:
                feasible_components = [c for c in self.components if c not in candidate]
            
            if len(feasible_components)==0:
                break
            else:
                # choose a feasible component
                if random.random()<=self.bias:
                    next_component = max(feasible_components)
                else:
                    next_component = inspyred.ec.selectors.fitness_proportionate_selection(random, feasible_components, {'num_selected':1})[0]
                candidate.append(next_component)
        return candidate

    def evaluator(self, candidates, args):
        # return the fitness values for the given candidate
        fitness = []
        fitness_function = args["fitness_function"]
        history_y = args["history_y"]
        for candidate in candidates:
            makespan = fitness_function([c.element for c in candidate],
                                        self.totalAvailableM,
                                        self.totalAvailableE,
                                        self.totalAvailableR,
                                        self.datasetFilePath)
            #fitness.append(self.max_makespan-makespan)
            fitness.append(1/makespan)
            history_y.append(makespan)
        return fitness
    
    def terminator(self, population, num_generations, num_evaluations, args):
        out_directory = args["output_directory"]
        if num_generations == args["max_generations"]:
            data = args["plot_data"]
            df = pd.DataFrame()
            df["generation"] = data[0]
            df["eval"] = data[1]
            df["average_fitness"] = data[2]
            df["median_fitness"] = data[3]
            df["best_fitness"] = data[4]
            df["worst_fitness"] = data[5]
            df.to_csv(f"{out_directory}/history_aco.csv", sep=",", index=False)
            print(f"Data successfully written to history_aco.csv")
        return num_generations == args["max_generations"]
    
    def observer(self, population, num_generations, num_evaluations, args):
        # current best individual
        best = max(population)

        # population size
        population_size = len(population)

        # store data of the plot fitness trend
        data = args["plot_data"]
        d0 = num_generations
        d1 = num_evaluations
        d2 = statistics.mean(args["history_y"])
        d3 = statistics.median(args["history_y"])
        d4 = max(args["history_y"])
        d5 = min(args["history_y"])
        data[0].append(d0)
        data[1].append(d1)
        data[2].append(d2)
        data[3].append(d3)
        data[4].append(d4)
        data[5].append(d5)

        # reset history_y for the next generation
        args["history_y"] = [] 

        print(f"OBSERVER\n[num generations:{num_generations}]\n[num evaluations:{num_evaluations}]\n[current best individual:{str(best.candidate)}]\n[population size:{population_size}]\n")
    
def read_arguments():
    parser = argparse.ArgumentParser(description="Adige hybrid flow shop scheduling optimizer.")
    
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")

    parser.add_argument("--out_dir", type=str, help="Output folder.")
    parser.add_argument('--no_runs', type=int, default=1, help='Number of runs.')

    parser.add_argument("--m", type=int, default=1, help="Resources of type M.")
    parser.add_argument("--e", type=int, default=1, help="Resources of type E.")
    parser.add_argument("--r", type=int, default=1, help="Resources of type R.")
    parser.add_argument("--dataset", type=str, default="data/d.csv", help="File path with the dataset.")
    parser.add_argument("--max_makespan", type=int, help="Max makespan. Used to shift from a minimization to a maximization problem.")

    parser.add_argument('--population_size', type=int, default=10, help='EA population size.')
    parser.add_argument('--max_generations', type=int, help='Generational budget.')

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
        #plt.plot(df['generation'], df['median_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Median Fitness')
        plt.plot(df['generation'], df['best_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
        plt.plot(df['generation'], df['worst_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Makespan')
        plt.title('Fitness Trend')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(f"{args['out_dir']}/aco.pdf")
        plt.savefig(f"{args['out_dir']}/aco.png")
    else:
        # create directory for saving results
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_path = f"{args['out_dir']}/{current_datetime}"
        os.makedirs(output_folder_path)

        # execution time csv file
        csv_execution_time_file_path = f"{output_folder_path}/exec_aco.csv"
        csv_execution_time_file = open(csv_execution_time_file_path, mode='w', newline='')
        csv_execution_time_writer = csv.writer(csv_execution_time_file)
        csv_execution_time_writer.writerow(["run", "time"])
        csv_execution_time_file.close()

        # read dataset jobs
        input_jobs: List[Job] = []
        input_df = pd.read_csv(args["dataset"])
        for index, row in input_df.iterrows():
            input_jobs.append(Job(order_id=row["order_id"],
                                  machine_type=row["machine_type"],
                                  date_basement_arrival=row["date_basement_arrival"],
                                  date_electrical_panel_arrival=row["date_electrical_panel_arrival"],
                                  date_delivery=row["date_delivery"]))

        # init anylogic model
        adige_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )

        # init design variable setup
        adige_setup = adige_model.get_jvm().adige.AdigeSetup()

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

        for r in range(args["no_runs"]):
            # create directory for saving results of the run
            output_folder_run_path = output_folder_path+"/"+str(r+1)
            os.makedirs(output_folder_run_path)

            # write a txt log file with algorithm configuration and other execution details
            log_file = open(f"{output_folder_run_path}/log.txt", 'w')
            log_file.write(f"algorithm: ant colony optimization\n")
            log_file.write(f"current date and time: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_file.write(f"max_generations: {args['max_generations']}\n")
            log_file.write(f"no_runs: {args['no_runs']}\n")
            log_file.write(f"m: {args['m']}\n")
            log_file.write(f"e: {args['e']}\n")
            log_file.write(f"r: {args['r']}\n")
            log_file.write(f"population_size: {args['population_size']}\n")
            log_file.write(f"\n===============\n")
            log_file.close()

            # init discrete optimization problem
            adige_problem = ACO_adige(jobs=input_jobs,
                                      totalAvailableM=args["m"],
                                      totalAvailableE=args["e"],
                                      totalAvailableR=args["r"],
                                      datasetFilePath=args["dataset"],
                                      max_makespan=args["max_makespan"])

            # run the optimizer
            start_time = time.time()
            #ac = inspyred.swarm.ACS(random=rng, components=adige_problem.components)
            ac = AdigeACS(random=rng, components=adige_problem.components)
            ac.observer = [adige_problem.observer]
            ac.terminator = [adige_problem.terminator]
            plot_data = [[], [], [], [], [], []]                                # fitness trend to plot
            plot_data[0] = []                                                   # generation number
            plot_data[1] = []                                                   # evaluation number
            plot_data[2] = []                                                   # average fitenss
            plot_data[3] = []                                                   # median fitness
            plot_data[4] = []                                                   # best fitness
            plot_data[5] = []                                                   # worst fitness
            final_pop = ac.evolve(  generator=adige_problem.constructor,        # the function to be used to generate candidate solutions
                                    evaluator=adige_problem.evaluator,          # the function to be used to evaluate candidate solutions
                                    pop_size=args["population_size"],           # the number of individuals in the population 
                                    max_generations=args["max_generations"],    # maximum generations
                                    maximize=adige_problem.maximize,            # boolean value stating use of maximization
                                    bounder=adige_problem.bounder,              # a function used to bound candidate solutions 
                                    fitness_function=simulation,                # fitness_function
                                    history_y=[],                               # keep track of individual fitness
                                    plot_data = plot_data,                      # data[0] generation number ; data[1] average fitenss ; data[2] median fitness ; data[3] best fitness ; data[4] worst fitness
                                    output_directory = output_folder_run_path   # output directory
                                )
            execution_time = time.time()-start_time
            # store execution time of the run
            csv_execution_time_file = open(csv_execution_time_file_path, mode='a', newline='')
            csv_execution_time_writer = csv.writer(csv_execution_time_file)
            csv_execution_time_writer.writerow([r, execution_time])
            csv_execution_time_file.close()
            log_file = open(f"{output_folder_run_path}/log.txt", 'a')
            log_file.write(f"total time: {execution_time}\n")
            log_file.close()
            print(f"total time: {time.time()-start_time}")
            
        # close model
        adige_model.close()