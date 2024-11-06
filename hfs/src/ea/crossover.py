import inspyred
import random
@inspyred.ec.variators.crossover
def ea_crossover(random, candidate1, candidate2, args):
    """Performs the order crossover (OX) on the parents.
    https://web.ist.utl.pt/adriano.simoes/tese/referencias/Michalewicz%20Z.%20Genetic%20Algorithms%20+%20Data%20Structures%20=%20Evolution%20Programs%20%283ed%29.PDF
    Args:
        random: The random number generator object.
        parents: A list of two parents to be crossed over.
        args: A dictionary of keyword arguments.
        
    Returns:
        A list containing the offspring.
    """
    size = len(candidate1)
    new_candidate1 = [-1] * size
    new_candidate2 = [-1] * size

    # choose two crossover points
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))

    # copy the segment from the first parent to the offspring
    new_candidate1[cxpoint1:cxpoint2] = candidate1[cxpoint1:cxpoint2]
    new_candidate2[cxpoint1:cxpoint2] = candidate2[cxpoint1:cxpoint2]

    # function to fill the remaining positions
    def fill_offspring(offspring, parent):
        current_offspring_pos = cxpoint2 % size
        current_parent_pos = cxpoint2 % size
        while current_offspring_pos != cxpoint1:
            if parent[current_parent_pos] not in offspring:
                offspring[current_offspring_pos] = parent[current_parent_pos]
                current_offspring_pos = (current_offspring_pos + 1) % size
            current_parent_pos = (current_parent_pos + 1) % size

    # fill the remaining positions with the order from the second parent
    fill_offspring(new_candidate1, candidate2)
    fill_offspring(new_candidate2, candidate1)
    
    return [new_candidate1, new_candidate2] 

if __name__ == '__main__':
    print("OX crossover proposed by Davis")
    rng = random.Random(42)
    p1 = [12, 0, 16, 3, 7, 13, 2, 6, 8, 10, 4, 5, 19, 1, 17, 18, 11, 9, 14, 15]
    p2 = [12, 7, 13, 8, 0, 19, 6, 9, 3, 10, 4, 5, 2, 1, 17, 18, 11, 16, 14, 15]
    ea_crossover(rng, [p1, p2], None)