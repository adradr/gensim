import logging
import pickle
from time import time
from tqdm import tqdm, trange
import gensim
from gensim import *
from gensim.Enviroment import SelectionCriterias

LOGGING_LEVEL = logging.INFO
MULTITHREADING_NTHREADS = 10

log = logging.getLogger('gensim')
logging.basicConfig(format="%(asctime)s - %(message)s")
log.setLevel(LOGGING_LEVEL)

if (__name__ == "__main__"):
    # Create enviroment
    log.info(f"Creating enviroment...")
    env = Enviroment.SimEnv(size=100,
                            population_size=200,
                            num_steps=50,
                            num_rounds=10,
                            gene_size=10,
                            num_int_neuron=3, mutation_probability=0.01,
                            selection_area_width_pct=0.1, criteria_type=SelectionCriterias.BOTH_SIDE,
                            multithreading=MULTITHREADING_NTHREADS)
    log.info(f"Enviroment created: {env.id}")

    # Iterate over steps
    start_time = time()
    log.info(f"Iterating enviroment steps: {env.id}")
    for i in range(env.num_steps):
        step_time_start = time()
        iter_time = round(step_time_start - start_time, 1)
        env.step()
        step_time_end = time()
        step_time = round(step_time_end - step_time_start, 1)
        log.info(
            f"Iterating progress: {env.num_steps}/{i} {step_time}s / {iter_time}s ---------------------------------")

    log.info(f"Iterating enviroment finished: {env.id}")

    # Generate animation
    log.info(f"Generating animation: {env.id}")
    env.generate_animation(10)

    # Saving env object to simulation dir
    save_path = env.sim_dir + 'enviroment.pickle'
    with open(save_path, 'wb') as output_dir:
        pickle.dump(env, output_dir)
