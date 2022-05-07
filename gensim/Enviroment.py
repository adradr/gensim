import os
import datetime
import logging
import sys
import uuid
import numpy as np
import pandas as pd
import imageio
import random
import copy
from time import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, wait

from gensim.Creatures import *

log = logging.getLogger('gensim')


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
num_round:           {num_rounds}/{num_round}"""

    return text


class SimEnv:

    def eval_env(self):
        # Store start time
        timer = {"start": time(), "step_start": time(),
                 "round_start": time()}

        # Iterate over enviroment
        for idx_round in range(self.num_rounds):
            for idx_step in range(self.num_steps):
                # Calculate step
                self.eval_step()
                timer["step"] = round(time() - timer["step_start"], 1)
                timer["step_start"] = time()
                timer["round"] = round(time() - timer["round_start"], 1)
                timer["iter"] = round(time() - timer["start"], 1)
                # Log progress
                log.debug(
                    f"Iterating progress (step/round/total): {self.num_steps}/{idx_step+1} {timer['step']}s / {self.num_rounds}/{idx_round+1} {timer['round']}s / {timer['iter']}s ---------------------------------")
            # Evaluate generation if num steps reached max
            self.eval_round()
            log.info(
                f"Iterating progress: {self.num_rounds}/{idx_round+1} {timer['round']}s / {timer['iter']}s ---------------------------------")

            timer["round_start"] = time()

    def eval_round(self):
        log.debug(f"Eval round:{self.num_round}")
        # Evaluate survivors
        survivors = self.selection.evaluate_survivors(
            self.creature_array)
        # Remove an element if uneven
        if len(survivors) % 2 == 1:
            survivors.pop()
            if len(survivors) == 0:
                log.info("No creatures survived. Exiting...")
                sys.exit(1)
        # Shuffle list randomly
        random.shuffle(survivors)

        # Match the survivors based randomly
        offsprings = []

        # Multithreading
        def threaded_loop():
            # Get an instance to work with from the survivor list
            choice = np.random.choice(survivors, 1)[0]
            cr = copy.deepcopy(choice)
            # Randomly select genes from their survivor parents
            cr.genome.set_random_genome_from_creatures(survivors)
            # Mutate if probability says so
            cr.genome.mutate_genome(self.mutation_probability)
            # Reinitialize offspring
            cr.reinit_offspring()
            # Append to offspring array
            offsprings.append(cr)

        if self.multithreading > 1:
            with ThreadPoolExecutor(self.multithreading) as executor:
                futures = [executor.submit(threaded_loop)
                           for i in range(self.population_size)]
                wait(futures)

        # Single threading
        elif self.multithreading == 1:
            for i in range(self.population_size):
                # Get an instance to work with from the survivor list
                choice = np.random.choice(survivors, 1)[0]
                cr = copy.deepcopy(choice)
                # Randomly select genes from their survivor parents
                cr.genome.set_random_genome_from_creatures(survivors)
                # Mutate if probability says so
                cr.genome.mutate_genome(self.mutation_probability)
                # Reinitialize offspring
                cr.reinit_offspring()
                # Append to offspring array
                offsprings.append(cr)

        # Set new population
        self.creature_array = offsprings
        # Reinit population
        self.init_population_locations()

        log.debug(
            f"Creature array locations - {[(x.X, x.Y) for x in self.creature_array]}")
        log.debug(
            f"Random locations array - {[tuple(x) for x in self.random_locations]}")

        # Increase round number
        self.num_round += 1
        # Reset step counter
        self.num_step = 1

    def eval_step(self):
        log.debug(f"Eval step:{self.num_step}")
        neuron_calc = NeuronCalculator()
        # Multithreading
        if self.multithreading > 1:
            with ThreadPoolExecutor(self.multithreading) as executor:
                # Calculate neurons
                futures = [executor.submit(neuron_calc.calc_neurons, cr)
                           for cr in self.creature_array]
                wait(futures)

                # Calculate action outputs
                futures = [executor.submit(neuron_calc.calc_action_outputs, cr)
                           for cr in self.creature_array]
                wait(futures)

                # Execute outputss on actions
                # Cannot execute them in parallel as they need to check if there is a creature where they would move
                # [] how to make this multithreading?
                [neuron_calc.execute_actions(cr) for cr in self.creature_array]

                # Reset neuron states
                futures = [executor.submit(neuron_calc.reset_neuron_states, cr)
                           for cr in self.creature_array]
                wait(futures)

        # Single threading
        elif self.multithreading == 1:
            for cr in self.creature_array:
                neuron_calc.calc_neurons(cr)
                neuron_calc.calc_action_outputs(cr)
                neuron_calc.execute_actions(cr)
                neuron_calc.reset_neuron_states(cr)

        # Saving image
        image = self.create_img()
        path = self.sim_subdir + \
            str(self.num_round) + '_' + str(self.num_step) + '.png'
        self.save_plot(path, image)
        # Calculate new occupied pixels
        self.occupied_pixels = self.calc_occupied_pixels()
        log.debug(
            f"Occupied pixels at step #{self.num_step} :\n{self.occupied_pixels}")
        # Increase step counter
        self.num_step += 1

    def init_population_locations(self):
        # Putting creatures to a unique random location
        self.set_random_locations(population_size=self.population_size)
        # Update new locations in creatures in their genome
        #[cr.genome.action.update_loc() for cr in self.creature_array]
        for cr in self.creature_array:
            cr.genome.action.update_loc()

        # Store occupied pixels
        self.occupied_pixels = self.calc_occupied_pixels()
        log.debug(
            f"Occupied pixels at init_population_locations:\n{self.occupied_pixels}")

    def set_random_locations(self, population_size: int):
        def generate_grid_locations(size: int):
            x = np.linspace(start=0, stop=size-1, num=size).astype(int)
            y = np.linspace(start=0, stop=size-1, num=size).astype(int)
            grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
            return grid

        grid = generate_grid_locations(self.X)
        np.random.shuffle(grid)
        self.random_locations = grid[:self.population_size]

        # Place creatures in new random locations
        for (loc, cr) in zip(self.random_locations, self.creature_array):
            cr.X = loc[0]
            cr.Y = loc[1]

    def calc_occupied_pixels(self):
        occupied_pixels = []
        for i in self.creature_array:
            occupied_pixels.append((i.X, i.Y))
        return occupied_pixels

    def create_img(self):
        def hex_to_rgb(value):
            """Return (red, green, blue) for the color given as #rrggbb."""
            value = value.lstrip('#')
            lv = len(value)
            tup = tuple(int(value[i:i + lv // 3], 16)
                        for i in range(0, lv, lv // 3))
            return np.array(tup)
        # Generate image with white pixels max alpha
        image = np.zeros((self.X, self.Y, 3), np.uint8)
        image.fill(255)
        # Fill with creatures painted black
        # rand_samples = np.random.randint(0, 100, size=(10, 2))
        for i in self.creature_array:
            image[i.Y, i.X] = hex_to_rgb(i.get_genome_hash())

        # Create second layer of selection criteria
        image_selection = np.zeros((self.X, self.Y, 3), np.uint8)
        # Fill alpha channel with low values
        image_selection.fill(255)
        for i in self.selection_pixels:
            image_selection[i[0], i[1]] = [105, 224, 137]
        self.selection_pixels_img = image_selection
        # Save image as result.png
        # filename = self.sim_subdir + str(self.num_step) + '.png'
        # cv2.imwrite(filename, image)
        # self.img_arr.append(image)
        return image

    def new_method(self, image_selection):
        image_selection[:, :, -1].fill(1)

    def save_plot(self, path: str, image: np.array):
        # Plotting single image
        plt.figure(figsize=(10, 5))
        plt.imshow(image, origin='lower', resample=False, alpha=1)
        plt.imshow(self.selection_pixels_img,
                   origin='lower', resample=False, alpha=0.4)
        text_str = get_text(self.id, self.X, self.population_size,
                            self.gene_size, self.num_int_neuron, self.mutation_probability,
                            self.num_steps, self.num_step,
                            self.num_rounds, self.num_round)
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='orange', alpha=1)
        # place a text box in upper left in axes coords
        plt.text((self.X + (self.X / 3)), 0, text_str, fontsize=14, bbox=props,
                 fontdict={"family": "monospace"})
        plt.savefig(path, dpi=100, facecolor='white', bbox_inches='tight')
        log.debug(f"Saving frame to:{path}")
        self.img_paths.append(path)
        plt.close()

    def save_animation(self, path: str, fps: int = 10):
        if path.split('.')[-1] == 'mp4':
            kargs = {'quality': 10, 'macro_block_size': None,
                     'ffmpeg_params': ['-s', '800x400']}
        if path.split('.')[-1] == 'gif':
            kargs = {}
        with imageio.get_writer(path, fps=fps, **kargs) as writer:
            for filename in self.img_paths:
                image = imageio.imread(filename)
                writer.append_data(image)
        # # Remove files
        # for filename in set(self.img_paths):
        #     os.remove(filename)

    def generate_animation(self, fps: int = 10):
        log.info('Generating simulation animation...')
        self.save_animation(self.anim_path_mp4, fps)
        self.save_animation(self.anim_path_gif, fps)
        log.info(f"Saved MP4 animation at {self.anim_path_mp4}")
        log.info(f"Saved GIF animation at {self.anim_path_gif}")

    def create_log(self):
        pass

    def __init__(self, size: int, population_size: int,
                 num_steps: int, num_rounds: int,
                 gene_size: int, num_int_neuron: int,
                 mutation_probability: float, selection_area_width_pct: int, criteria_type: str,
                 multithreading: int = 1):
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
        self.mutation_probability = mutation_probability

        # Init enviroment utils
        # [] implement logging - survivor rate, bio diversity
        self.log = pd.DataFrame()
        self.id = uuid.uuid4()
        self.multithreading = multithreading
        self.img_paths = []

        # Create folder for simulation
        now = datetime.datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        dir_path = 'simulations/' + date_time + '/'
        os.makedirs(dir_path, exist_ok=False)
        subdir_gene = dir_path + 'genomes/'
        os.makedirs(subdir_gene, exist_ok=False)
        subdir_path = dir_path + 'frames/'
        os.makedirs(subdir_path, exist_ok=False)
        self.sim_dir = dir_path
        self.sim_subdir = subdir_path
        self.sim_gendir = subdir_gene
        self.anim_path_mp4 = self.sim_dir + 'animation.mp4'
        self.anim_path_gif = self.sim_dir + 'animation.gif'

        # Init Creatures
        self.num_int_neuron = num_int_neuron
        self.gene_size = gene_size

        def create_cr_multi(gene_size: int, num_int_neuron: int):
            cr = Creature(env=self,
                          gene_size=gene_size,
                          num_int_neuron=num_int_neuron)
            cr.genome.action.update_loc()
            self.creature_array.append(cr)

        def save_cr_genome_img_multi(cr):
            cr.create_graph_img(view_img=False)

        # Generating creature multithread
        self.creature_array = []
        if self.multithreading > 1:
            with ThreadPoolExecutor(self.multithreading) as executor:
                # Generating creature
                futures = [executor.submit(create_cr_multi, gene_size=gene_size,
                                           num_int_neuron=num_int_neuron) for cr in range(self.population_size)]
                wait(futures)
                # Generating creature genome image
                futures = [executor.submit(save_cr_genome_img_multi, cr)
                           for cr in self.creature_array]
                wait(futures)

        elif self.multithreading == 1:
            for i in range(self.population_size):
                cr = Creature(env=self,
                              gene_size=gene_size,
                              num_int_neuron=num_int_neuron)
                cr.create_graph_img(view_img=False)
                cr.genome.action.update_loc()
                self.creature_array.append(cr)

        # # Putting creatures to a unique random location
        # self.set_random_locations(population_size=self.population_size)
        # # Update new locations in creatures
        # for cr in self.creature_array:
        #     cr.genome.action.update_loc()

        # # Store occupied pixels
        # self.occupied_pixels = self.calc_occupied_pixels()
        # log.debug(f"Occupied pixels:\n{self.occupied_pixels}")

        # Initialize population
        self.init_population_locations()
        log.info(f"Creatures generated.")

        # Init selection criteria
        self.criteria_type = criteria_type
        self.selection_area_width_pct = selection_area_width_pct
        self.selection = SelectionCriteria(
            self.criteria_type, self.selection_area_width_pct, self)
        self.selection_pixels = self.selection.calculate_area()

        # Saving first image
        image = self.create_img()
        path = self.sim_subdir + '1_0.png'
        self.save_plot(path, image)


class SelectionCriterias(Enum):
    LEFT_SIDE = 0
    RIGHT_SIDE = 1
    BOTH_SIDE = 2


class SelectionCriteria:
    def __init__(self, criteria_type: str, area_width_pct: int, enviroment: SimEnv):
        """SelectionCriteria initialization

        Args:
            criteria_type (str): type of the criteria to use
            area_width_pct (int): percentage of the total width for the selection area in pixels
        """
        self.criteria_type = criteria_type
        self.area_width_pct = area_width_pct
        self.env = enviroment
        self.size_x = self.env.X
        self.size_y = self.env.Y

    def calculate_area(self):
        if self.criteria_type == SelectionCriterias.LEFT_SIDE:
            pass
        if self.criteria_type == SelectionCriterias.RIGHT_SIDE:
            pass
        if self.criteria_type == SelectionCriterias.BOTH_SIDE:
            self.selection_pixels = []
            pct_width = round(self.size_x * self.area_width_pct, 0)
            #  Divide because two sides are split
            pct_width = int(round(pct_width / 2, 0))
            x_rightmost = self.size_x - pct_width
            x_leftmost = pct_width
            # Left side
            x_list = range(0, pct_width)
            y_list = range(0, self.size_y)
            for x in x_list:
                for y in y_list:
                    self.selection_pixels.append((x, y))
            # Left side
            x_list = range(x_rightmost, self.size_x)
            for x in x_list:
                for y in y_list:
                    self.selection_pixels.append((x, y))

        return self.selection_pixels

    def evaluate_survivors(self, creatures: list[Creature]):
        self.survivor_creatures_arr = []
        for i in creatures:
            if (i.X, i.Y) in self.selection_pixels:
                self.survivor_creatures_arr.append(i)
        return self.survivor_creatures_arr
