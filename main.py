import logging
import pickle
from tqdm import tqdm, trange
import gensim
from gensim import *
from gensim.Enviroment import SelectionCriterias

LOGGING_LEVEL = logging.INFO
MULTITHREADING = False
log = logging.getLogger('gensim')
logging.basicConfig(format="%(asctime)s - %(message)s")
log.setLevel(LOGGING_LEVEL)

if (__name__ == "__main__"):
    # Create enviroment
    log.info(f"Creating enviroment...")
    env = Enviroment.SimEnv(size=50,
                            population_size=5,
                            num_steps=5,
                            num_rounds=10,
                            gene_size=20,
                            num_int_neuron=3, mutation_probability=0.01,
                            selection_area_width_pct=0.1, criteria_type=SelectionCriterias.BOTH_SIDE,
                            multithreading=MULTITHREADING)
    log.info(f"Enviroment created: {env.id}")

    for cr in env.creature_array:
        log.info(f"{cr.id_short, cr.X, cr.Y}")

    # Iterate over steps
    log.info(f"Iterating enviroment steps: {env.id}")
    for i in range(env.num_steps):
        log.info(
            f"Iterating progress: {env.num_steps}/{i} ------------------------------------")
        env.step()
    log.info(f"Iterating enviroment finished: {env.id}")

    # Generate animation
    log.info(f"Generating animation: {env.id}")
    env.generate_animation(10)

    # Saving env object to simulation dir
    save_path = env.sim_dir + 'enviroment.pickle'
    with open(save_path, 'wb') as output_dir:
        pickle.dump(env, output_dir)
