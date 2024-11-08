import statistics

def ea_observer(population, num_generations, num_evaluations, args):
    # current best individual
    best = max(population)

    # population size
    population_size = len(population)

    # store data of the plot fitness trend
    data = args["plot_data"]
    data[0].append(num_generations)
    data[1].append(num_evaluations)
    data[2].append(statistics.mean(args["history_y"]))
    data[3].append(statistics.median(args["history_y"]))
    data[4].append(max(args["history_y"]))
    data[5].append(min(args["history_y"]))

    # reset history_y for the next generation
    args["history_y"] = [] 

    print(f"OBSERVER\n[num generations:{num_generations}]\n[num evaluations:{num_evaluations}]\n[current best individual:{best}]\n[population size:{population_size}]\n")