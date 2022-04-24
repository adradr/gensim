from tqdm import tqdm
from gensim import *


if (__name__ == '__main__'):
    # Create enviroment
    env = Enviroment.SimEnv(size=50,
                            population_size=10,
                            num_steps=200,
                            num_rounds=20,
                            gene_size=10,
                            num_int_neuron=3, mutation_rate=0.01)
    # Iterate over steps
    for i in tqdm(total=env.num_steps, iterable=range(env.num_steps)):
        env.step()

    # Generate animation
    env.generate_animation(10)
