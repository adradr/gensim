import logging
import pickle
from tqdm import tqdm, trange
import gensim
from gensim import *

LOGGING_LEVEL = logging.INFO
log = logging.getLogger('gensim')
logging.basicConfig(format="%(asctime)s - %(message)s")
log.setLevel(LOGGING_LEVEL)

if (__name__ == "__main__"):
    # Create enviroment
    log.info(f"Creating enviroment...")
    env = Enviroment.SimEnv(size=50,
                            population_size=250,
                            num_steps=50,
                            num_rounds=10,
                            gene_size=20,
                            num_int_neuron=3, mutation_rate=0.01)
    log.info(f"Enviroment created: {env.id}")

    print(len(env.creature_array))

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
