from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import argparse
import os
import pandas as pd
import random
import time

def read_arguments():
    parser = argparse.ArgumentParser(description="ABCD production optimizer.")
    
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")
    parser.add_argument("--out_dir", type=str, help="Output folder.")
    parser.add_argument('--no_runs', type=int, default=1, help='Number of runs.')

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
        plt.plot(range(len(df["y"])), df["y"].cummax(), marker='o', linestyle='-', linewidth=1, markersize=4, label='revenue')
        plt.xlabel('Trial')
        plt.ylabel('Revenue')
        plt.title('Fitness Trend')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(f"{args['out_dir']}/random.pdf")
        plt.savefig(f"{args['out_dir']}/random.png")
    else:
        # create directory for saving results
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_path = f"{args['out_dir']}/{current_datetime}"
        os.makedirs(output_folder_path)

        # execution time csv file
        csv_execution_time_file_path = f"{output_folder_path}/exec_random.csv"
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
            log_file.write(f"algorithm: random search\n")
            log_file.write(f"current date and time: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_file.write(f"n_trials: {args['n_trials']}\n")
            log_file.write(f"no_runs: {args['no_runs']}\n")
            log_file.write(f"\n===============\n")

            # history of evaluations
            history_x:List[List[int]] = []
            history_y:List[int] = []

            # run the optimizer
            best_y = -100000
            best_x = [random.randint(0,1) for _ in range(100)]
            start_time = time.time()
            for _ in range(args["n_trials"]):
                x = [random.randint(0,1) for _ in range(100)]
                y = simulation(x)
                if y>best_y:
                    best_y = y
                    best_x = x.copy()
                history_x.append(best_x.copy())
                history_y.append(int(y))
            execution_time = time.time()-start_time
            # store execution time of the run
            csv_execution_time_file = open(csv_execution_time_file_path, mode='a', newline='')
            csv_execution_time_writer = csv.writer(csv_execution_time_file)
            csv_execution_time_writer.writerow([r, execution_time])
            csv_execution_time_file.close()

            # print result
            print(f"Solution is {best_x} for a value of {best_y}")
            log_file.write(f"Solution is {best_x} for a value of {best_y}\n")

            # store fitness trend history in csv output file
            csv_file_path = f"{output_folder_run_path}/history_random.csv"
            csv_file = open(csv_file_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["x", "y"])
            for x, y in zip(history_x, history_y):
                csv_writer.writerow([str(x), y])
            csv_file.close()
            print(f"Data successfully written to history_random.csv")
            log_file.write(f"Data successfully written to history_random.csv\n")
            log_file.close()

        # close model
        abcd_model.close()