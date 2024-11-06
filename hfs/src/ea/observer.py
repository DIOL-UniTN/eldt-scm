import statistics

def ea_observer(population, num_generations, num_evaluations, args):
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

    print(f"OBSERVER\n[num generations:{num_generations}]\n[num evaluations:{num_evaluations}]\n[current best individual:{best}]\n[population size:{population_size}]\n")