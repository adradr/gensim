from gensim import *
from gensim.Enviroment import SelectionCriterias
import pickle
import os
import logging
from dotenv import load_dotenv, dotenv_values

# Preconfiguration
config = dotenv_values('.env')
log = logging.getLogger('gensim')
logging.basicConfig(format="%(asctime)s - %(message)s")
log.setLevel(eval(os.environ['LOGGING_LEVEL']))
for conf in config.items():
    log.info(conf)

# Main func
if (__name__ == "__main__"):

    # Create enviroment
    log.info(f"Creating enviroment...")
    env = Enviroment.SimEnv(size=int(os.environ['MAP_SIZE']),
                            population_size=int(os.environ['POPULATION_SIZE']),
                            num_steps=int(os.environ['NUM_STEPS']),
                            num_rounds=int(os.environ['NUM_GENERATIONS']),
                            gene_size=int(os.environ['GENE_SIZE']),
                            num_int_neuron=int(
                                os.environ['NUM_INTERNAL_NEURON']),
                            mutation_probability=float(
                                os.environ['MUTATION_RATE']),
                            selection_area_width_pct=float(
                                os.environ['SELECTION_AREA_SIZE']),
                            criteria_type=eval(
                                os.environ['SELECTION_CRITERIA_TYPE']),
                            multithreading=int(
                                os.environ['MULTITHREADING_NTHREADS']),
                            animation_fps=int(os.environ['ANIMATION_FPS']))
    log.info(f"Enviroment created: {env.id}")
    log.info(f"Disclaimer: Timing is only estimate, rounded and not accurate.")
    # Calculate simulation
    env.eval_env()
    log.info(f"Iterating enviroment finished: {env.id}")

    # Generate animation
    log.info(f"Generating animation: {env.id}")
    env.generate_animation(fps=10)

    # Saving env object to simulation dir
    save_path = env.sim_dir + 'enviroment.pickle'
    with open(save_path, 'wb') as output_dir:
        pickle.dump(env, output_dir)
