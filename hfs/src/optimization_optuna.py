from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from itertools import permutations
from datetime import datetime
import optuna
import matplotlib.pyplot as plt
import csv
import argparse
import os
import pandas as pd
import time

def read_arguments():
    parser = argparse.ArgumentParser(description="Adige hybrid flow shop scheduling optimizer.")
    
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")

    parser.add_argument("--n_trials", type=int, help="Number of optimization trials.") # default was 100
    parser.add_argument("--timeout", type=int, help="Stop study after the given number of second(s).")

    parser.add_argument("--out_dir", type=str, help="Output folder.")
    parser.add_argument('--no_runs', type=int, default=1, help='Number of runs.')

    parser.add_argument("--m", type=int, default=1, help="Resources of type M.")
    parser.add_argument("--e", type=int, default=1, help="Resources of type E.")
    parser.add_argument("--r", type=int, default=1, help="Resources of type R.")
    parser.add_argument("--dataset", type=str, default="data/d.csv", help="File path with the dataset.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()

    if args["input_csv"]:
        csv_file_path = args["input_csv"]
        df = pd.read_csv(csv_file_path)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(df["y"])), df["y"].cummin(), marker='o', linestyle='-', linewidth=1, markersize=4, label='makespan')
        plt.xlabel('Trial')
        plt.ylabel('Makespan')
        plt.title('Fitness Trend')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(f"{args['out_dir']}/optuna.pdf")
        plt.savefig(f"{args['out_dir']}/optuna.png")
    else:
        # create directory for saving results
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_path = f"{args['out_dir']}/{current_datetime}"
        os.makedirs(output_folder_path)

        adige_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )

        # init design variable setup
        adige_setup = adige_model.get_jvm().adige.AdigeSetup()

        # read dataset jobs
        input_jobs: List[int] = list()
        input_df = pd.read_csv(args["dataset"])
        input_jobs = input_df["order_id"].tolist()

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

        # setup and execute black box optimization model
        def objective(trial: optuna.trial.Trial,
                      jobs: List[int],
                      totalAvailableM: int,
                      totalAvailableE: int,
                      totalAvailableR: int,
                      datasetFilePath: str,
                      history_x: List[List[int]],
                      history_y: List[int]):
            priorities = [trial.suggest_int(f'{jobs[i]}', 0, len(jobs)) for i in range(len(jobs))]
            jobs_with_priorities = list(zip(jobs, priorities))
            sorted_jobs_with_priorities = sorted(jobs_with_priorities, key=lambda x: x[1], reverse=True)
            sorted_jobs = [job for job, _ in sorted_jobs_with_priorities]
            y = simulation(x=sorted_jobs,
                           totalAvailableM=totalAvailableM,
                           totalAvailableE=totalAvailableE,
                           totalAvailableR=totalAvailableR,
                           datasetFilePath=datasetFilePath)
            history_x.append(sorted_jobs.copy())
            history_y.append(int(y))
            return y

        # execution time csv file
        csv_execution_time_file_path = f"{output_folder_path}/exec_optuna.csv"
        csv_execution_time_file = open(csv_execution_time_file_path, mode='w', newline='')
        csv_execution_time_writer = csv.writer(csv_execution_time_file)
        csv_execution_time_writer.writerow(["run", "time"])
        csv_execution_time_file.close()

        for r in range(args["no_runs"]):
            # create directory for saving results of the run
            output_folder_run_path = output_folder_path+"/"+str(r+1)
            os.makedirs(output_folder_run_path)

            # write a txt log file with algorithm configuration and other execution details
            log_file = open(f"{output_folder_run_path}/log.txt", 'w')
            log_file.write(f"algorithm: optuna\n")
            log_file.write(f"current date and time: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_file.write(f"n_trials: {args['n_trials']}\n")
            log_file.write(f"timeout: {args['timeout']}\n")
            log_file.write(f"no_runs: {args['no_runs']}\n")
            log_file.write(f"m: {args['m']}\n")
            log_file.write(f"e: {args['e']}\n")
            log_file.write(f"r: {args['r']}\n")
            log_file.write(f"dataset: {args['dataset']}\n")
            log_file.write(f"\n===============\n")

            # history of evaluations
            history_x:List[List[int]] = []
            history_y:List[int] = []

            # run the optimizer
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=r+3))
            if args["n_trials"]:
                start_time = time.time()
                study.optimize(
                    lambda trial:   objective(  trial=trial,
                                                jobs=input_jobs,
                                                totalAvailableM=args["m"],
                                                totalAvailableE=args["e"],
                                                totalAvailableR=args["r"],
                                                datasetFilePath=args["dataset"],
                                                history_x=history_x,
                                                history_y=history_y),
                                    n_trials=args["n_trials"])
                execution_time = time.time()-start_time
                # store execution time of the run
                csv_execution_time_file = open(csv_execution_time_file_path, mode='a', newline='')
                csv_execution_time_writer = csv.writer(csv_execution_time_file)
                csv_execution_time_writer.writerow([r, execution_time])
                csv_execution_time_file.close()
            if args["timeout"]:
                study.optimize(
                    lambda trial:   objective(  trial=trial,
                                                jobs=input_jobs,
                                                totalAvailableM=args["m"],
                                                totalAvailableE=args["e"],
                                                totalAvailableR=args["r"],
                                                datasetFilePath=args["dataset"],
                                                history_x=history_x,
                                                history_y=history_y),
                                    timeout=args["timeout"])

            # print result
            best = study.best_params
            print(f"Solution is {best} for a value of {study.best_value}")
            log_file.write(f"Solution is {best} for a value of {study.best_value}\n")

            # store fitness trend history in csv output file
            csv_file_path = f"{output_folder_run_path}/history_optuna.csv"
            csv_file = open(csv_file_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["x", "y"])
            for x, y in zip(history_x, history_y):
                csv_writer.writerow([str(x), y])
            csv_file.close()
            print(f"Data successfully written to history_optuna.csv")
            log_file.write(f"Data successfully written to history_optuna.csv\n")
            log_file.close()
        
        # close model
        adige_model.close()