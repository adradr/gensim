import logging
import pickle
from tqdm import tqdm, trange
import gensim

LOGGING_LEVEL = logging.INFO

logging.basicConfig(format="%(asctime)s -  %(message)s")
log = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setStream(tqdm)  # <-- important
handler = log.addHandler(handler)
# Setting logging levels
log.setLevel(LOGGING_LEVEL)
for i in gensim.__all__:
    logging.getLogger(i).setLevel(LOGGING_LEVEL)

if (__name__ == "__main__"):
    # Create enviroment
    log.info(f"Creating enviroment...")
    env = gensim.Enviroment.SimEnv(size=100,
                                   population_size=100,
                                   num_steps=50,
                                   num_rounds=20,
                                   gene_size=10,
                                   num_int_neuron=3, mutation_rate=0.01)
    log.info(f"Enviroment created: {env.id}")

    # Iterate over steps
    log.info(f"Iterating enviroment steps: {env.id}")
    for i in trange(env.num_steps):
        env.step()
        #log.info(f"Size of env obj: {getsize(env)}")
    log.info(f"Iterating enviroment finished: {env.id}")

    # Generate animation
    log.info(f"Generating animation: {env.id}")
    env.generate_animation(10)

    # Saving env object to simulation dir
    save_path = env.sim_dir + 'enviroment.pickle'
    with open(save_path, 'wb') as output_dir:
        pickle.dump(env, output_dir)
