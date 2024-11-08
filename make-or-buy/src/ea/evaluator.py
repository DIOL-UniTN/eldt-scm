from tqdm import tqdm

def ea_evaluator(candidates, args):
    fitness_function = args["fitness_function"]
    history_y = args["history_y"]

    fitness = [None]*len(candidates)
    for index, a in tqdm(enumerate(candidates), total=len(candidates), desc=f"Candidates evaluation"):
        fitness[index] = fitness_function(a)
        history_y.append(int(fitness[index]))
    return fitness