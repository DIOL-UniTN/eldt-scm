from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import random
import numpy as np
import gymnasium as gym
import datetime
import os
import ray
import csv
import time
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
from alpypeopt import AnyLogicModel
from gymnasium import spaces
from functools import partial
from ast import literal_eval

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
                 seed: int):
        # machine type encoding
        '''
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
        self.machine_type_encoding: Dict[str, int] = {  # TO BE UPDATED WHEN THE NUMBER OF MACHINE CATEGORIES CHANGE
            "lt7": 0,
            "lt8": 1
        }
        # observation space
        # 1-D vector [machine_type, date_basement_arrival, date_electrical_panel_arrival, date_delivery, remaining_orders]
        low_bounds = np.array([0, 0, 0, 0, 0])
        high_bounds = np.array([len(self.machine_type_encoding)-1, max_date_basement_arrival, max_date_electrical_panel_arrival, max_delivery_date, num_jobs])
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.int64)

        # action space
        self.action_space = spaces.Discrete(n=num_jobs, seed=seed)

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
        #random.shuffle(self.input_jobs_ids)

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
        self.design_variable.append(action)

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
            makespan = self.simulation(sorted_job_ids)
            reward = self.max_makespan-makespan

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

def read_arguments():
    parser = argparse.ArgumentParser(description="Adige hybrid flow shop scheduling optimizer.")
    
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--out_dir", type=str, help="Output folder.")
    parser.add_argument('--no_runs', type=int, default=1, help='Number of runs.')
    parser.add_argument('--no_evaluations', type=int, default=5000, help='Number of evalutations.')

    parser.add_argument("--m", type=int, default=1, help="Resources of type M.")
    parser.add_argument("--e", type=int, default=1, help="Resources of type E.")
    parser.add_argument("--r", type=int, default=1, help="Resources of type R.")
    parser.add_argument("--num_machine_types", type=int, help="Number of machine types")
    parser.add_argument("--max_makespan", type=int, help="Max makespan. Used to avoid handling negative rewards.")
    parser.add_argument("--dataset", type=str, default="data/d.csv", help="File path with the dataset.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])
    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])

    if args["input_csv"]:
        # history of evaluations
        history_x:List[List[int]] = []
        history_y:List[int] = []
        history_evaluation:List[int] = []

        csv_file_path = args["input_csv"]
        csv_file = open(csv_file_path, mode='r', newline='')
        csv_reader = csv.reader(csv_file)
        next(csv_reader)    # skip the header
        current_best_y = float('inf')   # track the best y value encountered so far
        for row in csv_reader:
            x = literal_eval(row[0])  # convert the string representation of list back to a list
            eval = int(row[1])        # convert the string representation of integer back to an integer 
            y = int(float(row[2]))    # convert the string representation of integer back to an integer
            history_x.append(x)
            history_evaluation.append(eval)
            current_best_y = min(y, current_best_y)
            history_y.append(current_best_y)

        # plot fitness trent
        plt.figure(figsize=(10, 6))
        plt.plot(history_evaluation, history_y, marker='o', linestyle='-', color='b')
        plt.xlabel('Evaluation')
        plt.ylabel('Makespan')
        plt.title('Fitness Trend')
        plt.grid(True)
        #plt.show()
        plt.savefig(f"{args['out_dir']}/rl.pdf")
        plt.savefig(f"{args['out_dir']}/rl.png")
    else:
        # create directory for saving results
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder_path = f"{args['out_dir']}/{current_datetime}"
        os.makedirs(output_folder_path)

        # execution time csv file
        csv_execution_time_file_path = f"{output_folder_path}/exec_rl.csv"
        csv_execution_time_file = open(csv_execution_time_file_path, mode='w', newline='')
        csv_execution_time_writer = csv.writer(csv_execution_time_file)
        csv_execution_time_writer.writerow(["run", "time"])
        csv_execution_time_file.close()

        # read the input dataset
        input_df = pd.read_csv(args["dataset"])

        # register environment
        def env_creator(env_config):
            return AdigeEnv(
                max_date_basement_arrival=env_config['max_date_basement_arrival'],
                max_date_electrical_panel_arrival=env_config['max_date_electrical_panel_arrival'],
                max_delivery_date=env_config['max_delivery_date'],
                num_jobs=env_config['num_jobs'],
                totalAvailableM=env_config['totalAvailableM'],
                totalAvailableE=env_config['totalAvailableE'],
                totalAvailableR=env_config['totalAvailableR'],
                dataset_file_path=env_config['dataset_file_path'],
                max_makespan=env_config['max_makespan'],
                seed=env_config['seed']
            )

        for r in range(args["no_runs"]):
            # history of evaluations
            history_x:List[List[int]] = []
            history_y:List[int] = []
            history_evaluation:List[int] = []

            # create directory for saving results of the run
            output_folder_run_path = output_folder_path+"/"+str(r+1)
            os.makedirs(output_folder_run_path)

            # write a txt log file with algorithm configuration and other execution details
            log_file = open(f"{output_folder_run_path}/log.txt", 'w')
            log_file.write(f"algorithm: reinforcement learning\n")
            log_file.write(f"current date and time: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_file.write(f"no_evaluations: {args['no_evaluations']}\n")
            log_file.write(f"no_runs: {args['no_runs']}\n")
            log_file.write(f"m: {args['m']}\n")
            log_file.write(f"e: {args['e']}\n")
            log_file.write(f"r: {args['r']}\n")
            log_file.write(f"\n===============\n")
            log_file.close()

            # environment registration 
            register_env("AdigeEnv-v0", env_creator)

            # ray initialization
            context = ray.init()
            print(f"DASHBOARD: {context}")
            config = {
                "env": "AdigeEnv-v0",
                "env_config": {
                    "max_date_basement_arrival": input_df["date_basement_arrival"].max(),
                    "max_date_electrical_panel_arrival": input_df["date_electrical_panel_arrival"].max(),
                    "max_delivery_date": input_df["date_delivery"].max(),
                    "num_jobs": input_df.shape[0],
                    "totalAvailableM": args["m"],
                    "totalAvailableE": args["e"],
                    "totalAvailableR": args["r"],
                    "dataset_file_path": args["dataset"],
                    "max_makespan": args["max_makespan"],
                    "seed": args["random_seed"]
                },
                "framework": "torch",
                "num_workers": 1,
                "num_envs_per_worker": 1
            }

            algo = ppo.PPO(config=config)
            evaluations = 0
            i=0
            start_time = time.time()
            while evaluations < args["no_evaluations"]:
                result = algo.train()
                print(f"Iteration {i}:\nreward_min = {args['max_makespan']-result['env_runners']['episode_reward_min']}\nreward_mean = {args['max_makespan']-result['env_runners']['episode_reward_mean']}\nreward_max = {args['max_makespan']-result['env_runners']['episode_reward_max']}")
                evaluations += result['env_runners']["num_episodes"]
                i+=1
                history_x.append([1,1,1,1,1])
                history_evaluation.append(evaluations)
                history_y.append(args["max_makespan"]-result['env_runners']['episode_reward_max'])
            execution_time = time.time()-start_time
            # store execution time of the run
            csv_execution_time_file = open(csv_execution_time_file_path, mode='a', newline='')
            csv_execution_time_writer = csv.writer(csv_execution_time_file)
            csv_execution_time_writer.writerow([r, execution_time])
            csv_execution_time_file.close()

            # store fitness trend history in csv output file
            csv_file_path = f"{output_folder_run_path}/history_rl.csv"
            csv_file = open(csv_file_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["x", "eval", "y"])
            for x, eval, y in zip(history_x, history_evaluation, history_y):
                csv_writer.writerow([str(x), eval, y])
            csv_file.close()
            print(f"Data successfully written to history_rl.csv")
            
            ray.shutdown()