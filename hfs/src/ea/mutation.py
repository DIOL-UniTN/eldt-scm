import inspyred
import numpy as np
import random

@inspyred.ec.variators.mutator
def ea_mutation(rng, candidate, args):
    mutated_candidate = candidate.copy()
    if rng.random() < args["mutation_rate"]:
        # choose randomly how many times to swap genes
        number_of_mutations = rng.randint(1,args["mutation_swap_number"])
        for _ in range(number_of_mutations):
            pos1 = rng.randint(0, len(mutated_candidate)-1)
            pos2 = rng.randint(0, len(mutated_candidate)-1)
            while pos1==pos2:
                pos2 = rng.randint(0, len(mutated_candidate)-1)
            tmp = mutated_candidate[pos1]
            mutated_candidate[pos1] = mutated_candidate[pos2]
            mutated_candidate[pos2] = tmp
    return mutated_candidate