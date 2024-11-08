import pandas as pd

def ea_terminator(population, num_generations, num_evaluations, args):
   out_directory = args["output_directory"]
   if num_generations == args["generations_budget"]:
      data = args["plot_data"]
      df = pd.DataFrame()
      df["generation"] = data[0]
      df["eval"] = data[1]
      df["average_fitness"] = data[2]
      df["median_fitness"] = data[3]
      df["best_fitness"] = data[4]
      df["worst_fitness"] = data[5]
      df.to_csv(f"{out_directory}/history_ga.csv", sep=",", index=False)
   return num_generations == args["generations_budget"]