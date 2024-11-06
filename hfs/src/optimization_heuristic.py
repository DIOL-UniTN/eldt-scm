from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from itertools import permutations
import optuna
import matplotlib.pyplot as plt
import csv
import argparse
import pandas as pd

def read_arguments():
    parser = argparse.ArgumentParser(description="Adige hybrid flow shop scheduling optimizer.")
    
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    parser.add_argument("--out_dir", type=str, help="Output folder.")

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
        plt.plot(range(len(df["y"])), df["y"], marker='o', linestyle='-', linewidth=1, markersize=4, label='makespan')
        plt.xlabel('Trial')
        plt.ylabel('Makespan')
        plt.title('Fitness Trend')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(f"{args['out_dir']}/heuristic.pdf")
        plt.savefig(f"{args['out_dir']}/heuristic.png")
    else:
        adige_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )

        # init design variable setup
        adige_setup = adige_model.get_jvm().adige.AdigeSetup()

        # read dataset jobs
        heuristic_jobs: List[int] = list()
        input_df = pd.read_csv(args["dataset"])
        sorted_input_df = input_df.sort_values(by='date_delivery')  # sort the DataFrame by the date_delivery column
        heuristic_jobs = sorted_input_df['order_id'].tolist()

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
            print(f"x={x}")

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

        
        heuristic_result = simulation(x=heuristic_jobs,
                        totalAvailableM=args["m"],
                        totalAvailableE=args["e"],
                        totalAvailableR=args["r"],
                        datasetFilePath=args["dataset"])
        adige_model.close()

        # store fitness trend history in csv output file
        csv_file_path = f"{args['out_dir']}/history_heuristic.csv"
        csv_file = open(csv_file_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y"])
        csv_writer.writerow([str(heuristic_jobs), heuristic_result])
        csv_file.close()
        print(f"Data successfully written to history_heuristic.csv")