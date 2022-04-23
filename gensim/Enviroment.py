import os
import datetime
import logging
from operator import ge
import uuid
import numpy as np
import pandas as pd
import cv2
from gensim.Creatures import *

log = logging.getLogger(__name__)


class SelectionCriteria:
    def __init__(self, criteria_type: str, creature: Creature):
        """SelectionCriteria initialization

        Args:
            criteria_type (str): type of the criteria to use
            creature (Creature): Creature instance to calculate if it passes the selected criteria type
        """
        self.criteria_type = criteria_type
        self.creature = creature

    def eval_criteria(criteria_type: str, creature: Creature):
        if criteria_type == "LEFT_SIDE":
            pass
        if criteria_type == "RIGHT_SIDE":
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
        # Calculate required info
        len_sensory = len(SensoryNeurons)
        len_action = len(ActionNeurons)
        arr_sensory = [x.value for x in list(SensoryNeurons)]
        arr_action = [x.value for x in list(ActionNeurons)]
        arr_int_neurons = [
            x+len_sensory for x in np.arange(self.num_int_neuron)]

        # Iterating over creatures and evaluating their genes for a step in a round
        for i in self.creature_array:
            sensory = Sensory(i)
            action = Action(i)

            log.debug(i.last_dir, i.X, i.Y)

            for gene in i.gene_array:
                # [ 1., 12., 0., 1.,  0.09899215]
                # [ 0., 6., 1., 14., -2.21110532]
                # Sensory neurons output 0..1
                # Action neurons input tanh(sum(inputs)) -1..1
                # Action neurons output -4..4
                # Internal neurons input tanh(sum(inputs)) -1..1
                # Connection weights -5..5
                # If input source is action
                if gene[1] in arr_sensory:
                    input_val = getattr(
                        sensory, SensoryNeurons(gene[1]).name)()
                # If input source is internal neuron
                if gene[1] in arr_int_neurons:
                    input_val = i.int_neuron_state[gene[1]]

                log.debug(gene, i.X, i.Y, input_val)
                log.debug(SensoryNeurons(gene[1]).name if gene[1] in arr_sensory else gene[1], ActionNeurons(
                    gene[3]).name if gene[3] in arr_action else gene[3])

                # If output destination is action neuron
                if gene[3] in arr_action:
                    # getattr(action, ActionNeurons(gene(3)).name)(input_val)
                    i.action_neuron_state[gene[3]
                                          ] = i.action_neuron_state[gene[3]] + input_val
                # If output destination is internal neuron
                if gene[3] in arr_int_neurons:
                    i.int_neuron_state[gene[3]
                                       ] = i.int_neuron_state[gene[3]] + input_val

                log.debug(i.action_neuron_state)
                log.debug(i.int_neuron_state)

            # Calculate output neurons =tanh(sum(input)) = -1..1 for action and internals
            for f in i.int_neuron_state.items():
                inputs = np.array(i.int_neuron_state[f[0]])
                i.int_neuron_state[f[0]] = np.tanh(np.sum(inputs))

            for f in i.action_neuron_state.items():
                inputs = np.array(i.action_neuron_state[f[0]])
                i.action_neuron_state[f[0]] = np.tanh(np.sum(inputs))

            log.debug('Iteration over, executing action neurons')
            log.debug(i.action_neuron_state)
            log.debug(i.int_neuron_state)
            # Execute for all action neurons
            for h in i.action_neuron_state.items():
                if h[1]:
                    getattr(action, ActionNeurons(h[0]).name)(h[1])
                    log.debug(i.last_dir, i.X, i.Y)
                # [] need to multiply by synapse weights also
        i.action_neuron_state = dict.fromkeys(arr_action, 0)
        i.int_neuron_state = dict.fromkeys(arr_int_neurons, 0)
        log.debug(i.X, i.Y)
        # Generate new frame for step
        self.create_img()
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
        filename = self.sim_subdir + str(self.num_step)
        cv2.imwrite(filename, image)
        # self.img_arr.append(image)

    def save_animation(self):
        # create a video writer
        filename = self.sim_dir + '/grid_animation.gif'
        fps = 24
        # writer = cv2.VideoWriter()
        # writer = cv2.cvCreateVideoWriter(
        #     filename, -1, fps, cv2.Size(self.X, self.Y), is_color=1)
        # and write your frames in a loop if you want
        # for i in self.img_arr:
        #     cv2.cvWriteFrame(writer, i)

    def create_log(self):
        pass

    def __init__(self, size: int, population_size: int, num_steps: int, num_rounds: int, gene_size: int, num_int_neuron: int):
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
        self.num_step = 0
        self.round = 0
        self.max_round = num_rounds

        # Init population
        self.population_size = population_size
        self.init_population(self.population_size)

        # Init enviroment utils
        self.log = pd.DataFrame()
        self.id = uuid.uuid4()
        self.img_arr = []

        # Create folder for simulation
        now = datetime.datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S_")
        dir_path = 'simulations/' + date_time
        subdir_path = dir_path + '/frames/'
        os.makedirs(dir_path, exist_ok=False)
        os.makedirs(dir_path, exist_ok=False)
        self.sim_dir = dir_path
        self.sim_subdir = subdir_path

        # Init Creatures
        self.num_int_neuron = num_int_neuron
        self.gene_size = gene_size
        creature_array = []
        for i in self.random_locations:
            cr = Creature(env=self,
                          gene_size=gene_size, num_int_neuron=num_int_neuron)
            cr.X = i[0]
            cr.Y = i[1]
            creature_array.append(cr)
        self.creature_array = creature_array

        # Store occupied pixels
        self.occupied_pixels = self.calc_occupied_pixels()
