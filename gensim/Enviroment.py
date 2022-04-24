from gensim.Creatures import *
from tqdm import tqdm
import os
import datetime
import logging
import uuid
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_text(id, X, population_size,
             gene_size, num_int_neuron, mutation_rate,
             num_steps, num_step, num_rounds, num_round):
    text = f"""Simulation enviroment:
{id}

grid_size:           {X}
population_size:     {population_size}
gene_size:           {gene_size}
num_int_neuron:      {num_int_neuron}
mutation_rate:       {mutation_rate}
num_steps:           {num_steps}/{num_step}
max_round:"          {num_rounds}/{num_round}"""

    return text


class SelectionCriterias(Enum):
    LEFT_SIDE = 0
    RIGHT_SIDE = 1


class SelectionCriteria:
    def __init__(self, criteria_type: str, creature: Creature):
        """SelectionCriteria initialization

        Args:
            criteria_type (str): type of the criteria to use
            creature (Creature): Creature instance to calculate if it passes the selected criteria type
        """
        self.criteria_type = criteria_type
        self.creature = creature

    def eval_criteria(self):
        if self.criteria_type == SelectionCriterias.LEFT_SIDE:
            pass
        if self.criteria_type == SelectionCriterias.RIGHT_SIDE:
            pass


class SimEnv:
    def init_population(self, population_size: int):
        def generate_grid_locations(size: int):
            x = np.linspace(start=0, stop=size-1, num=size).astype(int)
            y = np.linspace(start=0, stop=size-1, num=size).astype(int)
            grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
            return grid

        def sample_grid_locations(array: np.array, num_population: int):
            selection = np.random.randint(
                0, array.shape[0], size=(num_population))
            mask = array[selection]
            return mask

        grid = generate_grid_locations(self.X)
        self.random_locations = sample_grid_locations(grid, population_size)

    def calc_occupied_pixels(self):
        occupied_pixels = []
        for i in self.creature_array:
            occupied_pixels.append((i.X, i.Y))
        return occupied_pixels

    def step(self):
        for cr in self.creature_array:
            log.info(cr, cr.X, cr.Y)
            # Calculate synapses' inputs
            cr.genome.calculate_synapses()
            # Calculate output neurons states
            cr.genome.calculate_outputs_neurons()
            # Execute neuron states
            cr.genome.execute_neuron_states()
        # Generate new frame for step
        image = self.create_img()
        path = self.sim_subdir + str(len(self.img_arr)) + '.png'
        self.save_plot(path, image)
        # Increase step counter
        self.num_step += 1

    def eval_round(self):
        pass

    def create_img(self):
        def hex_to_rgb(value):
            """Return (red, green, blue) for the color given as #rrggbb."""
            value = value.lstrip('#')
            lv = len(value)
            tup = tuple(int(value[i:i + lv // 3], 16)
                        for i in range(0, lv, lv // 3))
            return np.array(tup)
        # Generate image with OpenCV and place random dots
        image = np.zeros((self.X, self.Y, 3), np.uint8)
        image.fill(255)
        # Fill with creatures painted black
        # rand_samples = np.random.randint(0, 100, size=(10, 2))
        for i in self.creature_array:
            image[i.X, i.Y] = hex_to_rgb(i.get_genome_hash())
        # Save image as result.png
        filename = self.sim_subdir + str(self.num_step) + '.png'
        #cv2.imwrite(filename, image)
        self.img_arr.append(image)
        return image

    def save_plot(self, path: str, image: np.array):
        # Plotting single image
        plt.figure(figsize=(10, 5))
        plt.imshow(image, origin='lower', resample=False, alpha=1)
        text_str = get_text(self.id, self.X, self.population_size,
                            self.gene_size, self.num_int_neuron, self.mutation_rate,
                            self.num_steps, self.num_step,
                            self.num_rounds, self.num_round)
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='orange', alpha=1)
        # place a text box in upper left in axes coords
        plt.text((self.X + (self.X / 3)), 0, text_str, fontsize=14, bbox=props,
                 fontdict={"family": "monospace"})
        plt.savefig(path, dpi=100, facecolor='white', bbox_inches='tight')
        log.info(path)
        self.img_paths.append(path)
        plt.close()

    def save_animation(self, path: str, fps: int = 10):
        kargs = {'quality': 10, 'macro_block_size': None,
                 'ffmpeg_params': ['-s', '800x400']}
        with imageio.get_writer(self.anim_path, fps=fps, **kargs) as writer:
            for filename in self.img_paths:
                image = imageio.imread(filename)
                writer.append_data(image)
        # # Remove files
        # for filename in set(self.img_paths):
        #     os.remove(filename)

    def generate_animation(self, fps: int = 10):
        log.info('Generating enviroment animation...')
        self.save_animation(self.anim_path, fps)
        log.info(f"Saved animation at {self.anim_path}")

    def create_log(self):
        pass

    def __init__(self, size: int, population_size: int,
                 num_steps: int, num_rounds: int,
                 gene_size: int, num_int_neuron: int,
                 mutation_rate: int):
        """Enviroment initialization

        Args:
            size (int): size of the grid in pixels size * size 
            population_size (int): number of creatures to initialize
            num_steps (int): number of steps in a round, each step is an action for a creature
        """
        # Init enviroment space
        self.area = size * size
        self.X = size
        self.Y = size
        self.num_steps = num_steps
        self.num_step = 1
        self.num_round = 1
        self.num_rounds = num_rounds

        # Init population
        self.population_size = population_size
        self.init_population(self.population_size)
        self.mutation_rate = mutation_rate

        # Init enviroment utils
        self.log = pd.DataFrame()
        self.id = uuid.uuid4()
        self.img_arr = []
        self.img_paths = []

        # Create folder for simulation
        now = datetime.datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        dir_path = 'simulations/' + date_time + '/'
        subdir_path = dir_path + 'frames/'
        os.makedirs(dir_path, exist_ok=False)
        os.makedirs(subdir_path, exist_ok=False)
        self.sim_dir = dir_path
        self.sim_subdir = subdir_path
        self.anim_path = self.sim_dir + 'animation.mp4'

        # Init Creatures
        self.num_int_neuron = num_int_neuron
        self.gene_size = gene_size
        creature_array = []
        for i in self.random_locations:
            cr = Creature(env=self,
                          gene_size=gene_size,
                          num_int_neuron=num_int_neuron)
            cr.X = i[0]
            cr.Y = i[1]
            creature_array.append(cr)
        self.creature_array = creature_array

        # Store occupied pixels
        self.occupied_pixels = self.calc_occupied_pixels()
        log.debug("Occupied pixels:", self.occupied_pixels)
