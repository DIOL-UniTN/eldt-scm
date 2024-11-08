from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import pandas as pd
import argparse
import random
import os
import time

import inspyred
from ea.observer import ea_observer
from ea.generator import ea_generator
from ea.evaluator import ea_evaluator
from ea.terminator import ea_terminator


def read_arguments():
    parser = argparse.ArgumentParser(description="ABCD production optimizer.")
    
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    
    parser.add_argument("--out_dir", type=str, help="Output folder.")
    parser.add_argument('--no_runs', type=int, default=1, help='Number of runs.')

    parser.add_argument('--population_size', type=int, default=6, help='EA population size.')
    parser.add_argument('--offspring_size', type=int, default=6, help='EA offspring size.')
    parser.add_argument('--max_generations', type=int, default=100, help='Generational budget.')
    parser.add_argument('--mutation_rate', type=float, default=1.0, help='EA mutation rate.')
    parser.add_argument('--crossover_rate', type=float, default=1.0, help='EA crossover rate.')
    parser.add_argument('--num_elites', type=int, default=1, help='EA number of elite individuals.')

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
        plt.plot(df['generation'], df['median_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Median Fitness')
        plt.plot(df['generation'], df['best_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
        plt.plot(df['generation'], df['worst_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Revenue')
        plt.title('Fitness Trend')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(f"{args['out_dir']}/ga.pdf")
        plt.savefig(f"{args['out_dir']}/ga.png")
    else:
        # create directory for saving results
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_path = f"{args['out_dir']}/{current_datetime}"
        os.makedirs(output_folder_path)

        # execution time csv file
        csv_execution_time_file_path = f"{output_folder_path}/exec_ga.csv"
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

        for r in range(args["no_runs"]):
            # create directory for saving results of the run
            output_folder_run_path = output_folder_path+"/"+str(r+1)
            os.makedirs(output_folder_run_path)

            # write a txt log file with algorithm configuration and other execution details
            log_file = open(f"{output_folder_run_path}/log.txt", 'w')
            log_file.write(f"algorithm: genetic algorithm\n")
            log_file.write(f"current date and time: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_file.write(f"max_generations: {args['max_generations']}\n")
            log_file.write(f"no_runs: {args['no_runs']}\n")
            log_file.write(f"population_size: {args['population_size']}\n")
            log_file.write(f"offspring_size: {args['offspring_size']}\n")
            log_file.write(f"mutation_rate: {args['mutation_rate']}\n")
            log_file.write(f"crossover_rate: {args['crossover_rate']}\n")
            log_file.write(f"num_elites: {args['num_elites']}\n")
            log_file.write(f"\n===============\n")
            log_file.close()

            # run the optimizer
            # default: rank selection,
            # default: n-point crossover,
            # default: bit-flip mutation,
            # default: generational replacement
            ea = inspyred.ec.GA(rng)
            ea.observer = [ea_observer]
            ea.terminator = ea_terminator
            plot_data = [[], [], [], [], [], []]                                    # fitness trend to plot
            plot_data[0] = []                                                       # generation number
            plot_data[1] = []                                                       # evaluation number
            plot_data[2] = []                                                       # average fitenss
            plot_data[3] = []                                                       # median fitness
            plot_data[4] = []                                                       # best fitness
            plot_data[5] = []                                                       # worst fitness
            start_time = time.time()
            final_pop = ea.evolve(  generator=ea_generator,                         # the function to be used to generate candidate solutions
                                    evaluator=ea_evaluator,                         # the function to be used to evaluate candidate solutions
                                    pop_size=args["population_size"],               # the number of Individuals in the population 
                                    num_selected=args["offspring_size"],            # offspring of the EA
                                    generations_budget=args["max_generations"],     # maximum generations
                                    maximize=True,                                  # boolean value stating use of maximization
                                    bounder=inspyred.ec.DiscreteBounder([0, 1]),    # a function used to bound candidate solutions 
                                    num_elites=args["num_elites"],                  # number of elites to consider
                                    fitness_function=simulation,                    # fitness_function
                                    history_y=[],                                   # keep track of individual fitness
                                    plot_data = plot_data,                          # data[0] generation number ; data[1] average fitenss ; data[2] median fitness ; data[3] best fitness ; data[4] worst fitness
                                    output_directory = output_folder_run_path       # directory folder where to store the final results
                                )
            execution_time = time.time()-start_time
            # store execution time of the run
            csv_execution_time_file = open(csv_execution_time_file_path, mode='a', newline='')
            csv_execution_time_writer = csv.writer(csv_execution_time_file)
            csv_execution_time_writer.writerow([r, execution_time])
            csv_execution_time_file.close()
        # close model
        abcd_model.close()
