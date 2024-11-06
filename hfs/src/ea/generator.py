import inspyred
@inspyred.ec.generators.diversify # decorator that makes it impossible to generate copies
def ea_generator(random, args):
    job_ids = args["job_ids"].copy()
    random.shuffle(job_ids)
    return job_ids