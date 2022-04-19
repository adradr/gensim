import logging
from operator import ge
import uuid
import numpy as np
import pandas as pd
import cv2
from gensim.Creatures import Creature

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


class Enviroment:
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

    def sim_step(self):
        pass

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
        image = np.zeros((100, 100, 3), np.uint8)
        image.fill(255)
        # Fill with creatures painted black
        #rand_samples = np.random.randint(0, 100, size=(10, 2))
        for i in self.creature_array:
            image[i.X, i.Y] = hex_to_rgb(i.get_gene_hash())
        # Save image as result.png
        cv2.imwrite("result.png", image)

    def create_log(self):
        pass

    def __init__(self, size: int, population_size: int, num_steps: int, gene_size: int, num_int_neuron: int):
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

        # Init population
        self.population_size = population_size
        self.init_population(self.population_size)

        # Init enviroment utils
        self.log = pd.DataFrame()
        self.id = uuid.uuid4()
        self.round = 0

        # Init Creatures
        creature_array = []
        for i in self.random_locations:
            cr = Creature(gene_size=gene_size, num_int_neuron=num_int_neuron)
            cr.X = i[0]
            cr.Y = i[1]
            creature_array.append(cr)
        self.creature_array = creature_array
