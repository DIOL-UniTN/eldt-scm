import inspyred
@inspyred.ec.generators.diversify # decorator that makes it impossible to generate copies
def ea_generator(random, args):
    return [random.randint(0,1) for _ in range(100)]