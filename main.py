import logging
import pickle
from time import time
from gensim import *
from gensim.Enviroment import SelectionCriterias

# -------- SETTINGS --------
LOGGING_LEVEL = logging.INFO
MAP_SIZE = 25
POPULATION_SIZE = 200
NUM_STEPS = 100
NUM_GENERATIONS = 20
GENE_SIZE = 12
NUM_INTERNAL_NEURON = 3
MUTATION_RATE = 0.05
SELECTION_AREA_SIZE = 0.2
SELECTION_CRITERIA_TYPE = SelectionCriterias.BOTH_SIDE
MULTITHREADING_NTHREADS = 200
# --------------------------

log = logging.getLogger('gensim')
logging.basicConfig(format="%(asctime)s - %(message)s")
log.setLevel(LOGGING_LEVEL)

if (__name__ == "__main__"):
    # Create enviroment
    log.info(f"Creating enviroment...")
    env = Enviroment.SimEnv(size=MAP_SIZE,
                            population_size=POPULATION_SIZE,
                            num_steps=NUM_STEPS,
                            num_rounds=NUM_GENERATIONS,
                            gene_size=GENE_SIZE,
                            num_int_neuron=NUM_INTERNAL_NEURON,
                            mutation_probability=MUTATION_RATE,
                            selection_area_width_pct=SELECTION_AREA_SIZE,
                            criteria_type=SELECTION_CRITERIA_TYPE,
                            multithreading=MULTITHREADING_NTHREADS)
    log.info(f"Enviroment created: {env.id}")
    # Calculate simulation
    env.eval_env()
    log.info(f"Iterating enviroment finished: {env.id}")

    # Generate animation
    log.info(f"Generating animation: {env.id}")
    env.generate_animation(10)

    # Saving env object to simulation dir
    save_path = env.sim_dir + 'enviroment.pickle'
    with open(save_path, 'wb') as output_dir:
        pickle.dump(env, output_dir)
