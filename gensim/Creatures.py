from operator import inv
import sys
import os
import logging
import math
import uuid
import hashlib
import graphviz
import numpy as np
from enum import Enum

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def generate_pixels_around(grid_size: int, current_location: tuple):
    # Getting a nullpoint for the grid_size * grid_size search grid
    half_grid_size = math.ceil(grid_size/2)
    null_point = [x-half_grid_size for x in current_location]
    # Creating the pixel locations in the pixels around creature
    density_loc_arr = []
    for i in range(1, grid_size + 1):
        for f in range(1, grid_size + 1):
            new_loc = (null_point[0] + f, null_point[1] + i)
            density_loc_arr.append(new_loc)
    return density_loc_arr


class Creature:
    def process_enviroment(self):
        pass

    def mutate(self):
        pass

    def generate_genes():
        pass

    def generate_synapse():
        pass

    def create_graph_img(self, view_img=False):
        genome_hash = self.get_genome_hash()
        save_path = f"{self.sim_id}/{genome_hash}"
        dot = graphviz.Digraph(comment=genome_hash)
        dot.attr(ratio='auto', size='30',
                 fontname="Helvetica", label=f"Genome hash: {genome_hash}\n{self.sim_id}")
        for idx, i in enumerate(self.gene_array):
            # Node values
            a = list(SensoryNeurons)[np.int(i[1])
                                     ].name if i[0] == 0 else 'N' + str(np.int(i[1]))
            b = list(ActionNeurons)[np.int(i[3])
                                    ].name if i[2] == 0 else 'N' + str(np.int(i[3]))
            # Orange if sensory, gray if internal
            id_1_color = "#FFA500" if i[0] == 0 else "#a5a0a8"
            # Blue if action, gray if internal
            id_2_color = "#b17cd6" if i[2] == 0 else "#a5a0a8"

            dot.node(a, style='filled', fillcolor=id_1_color)
            dot.node(b, style='filled', fillcolor=id_2_color)
            rounded_weight = round(i[4], 2)
            edge_color = "#2117b0" if rounded_weight > 0 else "#b01726"
            dot.edge(a, b, str(rounded_weight), {
                     "penwidth": str(abs(rounded_weight+2)), "color": edge_color})
            # dot.unflatten(stagger=5)
        dot.render(save_path, view=view_img, format="png")
        # Remove dotfile
        os.remove(save_path)

    def get_genome_hash(self):
        hash = hashlib.sha1(self.gene_array).hexdigest()
        return str(hash)[:6]

    def __init__(self, env, gene_size: int, num_int_neuron: int):
        # Get genome
        genome = Genome(
            gene_size=gene_size, num_int_neuron=num_int_neuron)
        self.gene_array = genome.genome
        self.id = uuid.uuid4()
        self.env = env
        self.sim_id = env.sim_dir
        log.info(
            f"Creature {str(self.id)[:8]} neuron array:\n{self.gene_array}")
        # Location data
        self.X = 0
        self.Y = 0
        self.last_dir = Directions[Directions._member_names_[
            np.random.randint(len(Directions))]]
        # Setting oscillator with a random init frequency
        self.oscillator = Oscillator(np.random.uniform(1, 5))
        # Setting init attributes
        self.age = 0
        self.sex = np.random.randint(2)  #  0 - male, 1 - female


class Oscillator:
    def get_value_at_step(self, step):
        return self.signal[step]

    def set_period(self, freq: float):
        assert 5 >= freq >= 1, "Use value between 5 and 1"
        self.freq = freq
        time = np.arange(0, 100, 0.1)
        signal = np.sin(time / self.freq) + 1
        self.signal = signal

    def __init__(self, freq: float):
        self.freq = freq
        self.set_period(self.freq)
        self.max_value = max(self.signal)

# 1. Sx - location on the X axis
# 2. Sy - location on the Y axis
# 3. Dn - distance from north
# 4. Ds - distance from south
# 5. Dw - distance from west
# 6. De - distance from east
# 7. Da - density around: how many individuals are around (8 pixels)
# 8. Va - view ahead forward: is there an individual in the next 3 pixels
# 9. Ph - pheromons detected around (5x5 pixels)
# 10. Se - sex
# 11. Ag - age
# 12. Os - internal oscillator signal


class Sensory:
    def map_0_1(self, value: int, max_value: int):
        return value / max_value

    def get_x_loc(self):
        self.x_loc = self.creature.X
        return self.map_0_1(self.x_loc, self.x_loc_max)

    def get_y_loc(self):
        self.y_loc = self.creature.Y
        return self.map_0_1(self.y_loc, self.y_loc_max)

    def get_dst_north(self):
        self.dst_north = self.creature.env.Y - self.creature.Y
        return self.map_0_1(self.dst_north, self.y_loc_max, )

    def get_dst_south(self):
        self.dst_south = self.creature.Y
        return self.map_0_1(self.dst_south, self.y_loc_max)

    def get_dst_west(self):
        self.dst_west = self.creature.env.Y - self.creature.X
        return self.map_0_1(self.dst_west, self.x_loc_max)

    def get_dst_east(self):
        self.dst_east = self.creature.X
        return self.map_0_1(self.dst_east, self.x_loc_max)

    def get_density_around(self):
        grid_size = 5 * 5
        # Getting a nullpoint for the 5x5 search grid
        null_point = [x-2 for x in self.creature_loc]
        # Creating the pixel locations in 5 pixels around creature
        density_loc_arr = generate_pixels_around(5, self.creature_loc)
        # Calculating the unique number of difference between the total grid and proximity field
        total_arr = density_loc_arr + self.pixel_arr
        density = len(list(set(total_arr))) - len(self.pixel_arr)
        self.density = grid_size - density - 1  #  -1 for the creature location
        return self.map_0_1(self.density, grid_size-1)

    def get_view_forward(self, number_pixel_ahead: int = 3):
        locations_ahead = []
        # Calculate pixel locations ahead by number_pixel_ahead
        for i in range(number_pixel_ahead):
            self.creature_loc = np.add(
                tuple(self.direction), self.creature_loc)
            locations_ahead.append(tuple(self.creature_loc))
        # Search array of ahead locations if there are other creatures
        total_arr = locations_ahead + self.pixel_arr
        self.pixels_ahead = len(self.pixel_arr)-len(list(set(total_arr)))
        return self.map_0_1(self.pixels_ahead, number_pixel_ahead)

    def get_pheromons_around(self):
        # [] need to implement
        return 0

    def get_sex(self):
        self.sex = self.creature.sex
        return self.sex

    def get_age(self):
        self.age = self.creature.age
        return self.map_0_1(self.age, self.env_max_age)

    def get_oscillator(self):
        self.oscillator = self.creature.oscillator.get_value_at_step(
            self.creature.env.step)
        print(self.oscillator)
        return self.map_0_1(self.oscillator, self.creature.oscillator.get_max_value())

    def __init__(self, creature: Creature):
        self.creature = creature
        self.y_loc_max = self.creature.env.Y
        self.x_loc_max = self.creature.env.X
        self.creature_loc = (self.creature.X, self.creature.Y)
        self.pixel_arr = self.creature.env.occupied_pixels
        self.direction = self.creature.last_dir.value
        self.env_max_age = self.creature.env.max_round


class Directions(Enum):
    EAST = (1, 0)
    WEST = (-1, 0)
    NORTH = (0, 1)
    SOUTH = (0, -1)


class OppositeDirections(Enum):
    EAST = Directions.WEST
    WEST = Directions.EAST
    NORTH = Directions.SOUTH
    SOUTH = Directions.NORTH


class RightDirections(Enum):
    EAST = Directions.SOUTH
    WEST = Directions.NORTH
    NORTH = Directions.EAST
    SOUTH = Directions.WEST


class LeftDirections(Enum):
    EAST = Directions.NORTH
    WEST = Directions.SOUTH
    NORTH = Directions.WEST
    SOUTH = Directions.EAST


# 1. Mfr - move forward (previous direction)
# 2. Mrn - move random
# 3. Mlr - move left/right
# 4. Mew - move east/west
# 5. Mns - move north/south
# 6. So - set oscillator period
# 7. Ep - emit pheromone

class Action:
    def __init__(self, creature: Creature):
        self.creature = creature
        self.loc = [self.creature.X, self.creature.Y]
        self.last_dir = self.creature.last_dir
        self.dir_values = [e.value for e in Directions]
        self.dir_keys = [e.name for e in Directions]

    def update_last_direction(self, last_loc: tuple, new_loc: tuple):
        last_dir = tuple([x-last_loc[idx] for idx, x in enumerate(new_loc)])
        try:
            idx = self.dir_values.index(last_dir)
            self.last_dir = Directions[self.dir_keys[idx]]
            self.creature.last_dir = self.last_dir
        except:
            return

    def move(self, direction: Directions, value: float = 1):
        new_loc = [(x + direction.value[idx]) * value
                   for idx, x in enumerate(self.loc)]
        # Check so neither coordinates cannot go below zero, else set to zero
        new_loc = [0 if x < 0 else x for x in new_loc]
        self.update_last_direction(last_loc=self.loc, new_loc=new_loc)
        self.loc = new_loc
        return new_loc

    def mfr(self, value: float):
        direction = self.last_dir if value > 0 else OppositeDirections[self.last_dir.name].value
        new_loc = self.move(direction=direction, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def mlr(self, value: float):
        direction = RightDirections[self.last_dir.name].value if value > 0 else LeftDirections[self.last_dir.name].value
        new_loc = self.move(direction=direction, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def mew(self, value: float):
        direction = Directions.EAST if value > 0 else Directions.WEST
        new_loc = self.move(direction=direction, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def mns(self, value: float):
        direction = Directions.NORTH if value > 0 else Directions.SOUTH
        new_loc = self.move(direction=direction, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def mrn(self):
        rand_dir = Directions[np.random.choice(Directions._member_names_)]
        new_loc = self.move(rand_dir)
        [self.creature.X, self.creature.Y] = new_loc

    def so(self, period):
        self.creature.oscillator.set_period(period)

    def ep(self):
        pass


class GeneType(Enum):
    INTERNAL = 1
    NORMAL = 0


class Genome:
    def get_gene_hash(self, neuron):
        hash = hashlib.sha1(neuron).hexdigest()
        return str(hash)[:6]

    def update_input(self, inputs: np.array):
        self.inputs = inputs

    def calculate_synapse(self):
        pass

    def generate_int_neuron_list(self, num_int_neuron: int):
        len_action = len(SensoryNeurons)
        return np.arange(len_action, len_action + num_int_neuron)

    def generate_gene(self, int_neuron_arr: list):
        # [source_type][from_neuron_id][destination_type][to_neuron_id][synapse_weight]

        # Calculate max lenghts
        max_sensory_len = len(SensoryNeurons)
        max_action_len = len(ActionNeurons)

        source_type = np.random.randint(2)
        from_neuron_id = np.random.randint(
            max_sensory_len) if source_type == 0 else int(np.random.choice(int_neuron_arr, 1))
        destination_type = np.random.randint(2)
        to_neuron_id = np.random.randint(
            max_action_len) if destination_type == 0 else int(np.random.choice(int_neuron_arr, 1))

        # Setting synapse weight between 1 and 5
        synapse_weight = np.random.uniform(low=-5, high=5)

        array = np.array([source_type, from_neuron_id,
                         destination_type, to_neuron_id, synapse_weight])

        # Getting hash
        self.gene_hash.append(self.get_gene_hash(array))
        return array

    def generate_full_genome(self, gene_size: int, num_int_neuron: int):
        # Get internal neuron list
        int_neuron_arr = self.generate_int_neuron_list(num_int_neuron)
        # Generate gene pool for Creature
        gene_array = []
        for i in range(gene_size):
            gene_array.append(self.generate_gene(int_neuron_arr))
        return np.array(gene_array)

    def __init__(self, gene_size: int, num_int_neuron: int):
        self.gene_hash = []
        self.genome = self.generate_full_genome(
            gene_size=gene_size, num_int_neuron=num_int_neuron)
        # self.inputs = np.empty(self.neuron_shape, dtype=object)


class SensoryNeurons(Enum):
    Sx = 0
    Sy = 1
    Dn = 2
    Ds = 3
    Dw = 4
    De = 5
    Da = 6
    Va = 7
    Ph = 8
    Se = 9
    Ag = 10
    Os = 11


class ActionNeurons(Enum):
    Mfr = 0
    Mrn = 1
    Mlr = 2
    Mew = 3
    Mns = 4
    So = 5
    Ep = 6
