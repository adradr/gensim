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
    def create_graph_img(self, view_img=False):
        genome_hash = self.get_genome_hash()
        save_path = f"{self.sim_id}/{genome_hash}"
        dot = graphviz.Digraph(comment=genome_hash)
        dot.attr(ratio='auto', size='30',
                 fontname="Helvetica", label=f"Genome hash: {genome_hash}\n{self.sim_id}")
        for idx, i in enumerate(self.gene_array):
            # Node values
            idx_sensory = SensoryNeurons
            a = str(int(i[1])) + '.' + list(SensoryNeurons)[
                np.int(i[1])].name if i[0] == 0 else 'N' + str(np.int(i[1]))
            b = str(int(i[3])) + '.' + list(ActionNeurons)[np.int(i[3])
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
        self.gene_size = gene_size
        self.num_int_neuron = num_int_neuron
        genome = Genome(
            gene_size=gene_size, num_int_neuron=num_int_neuron, creature=self)
        self.genome = genome
        self.gene_array = genome.genome

        # Calculate required info
        len_sensory = len(SensoryNeurons)
        len_action = len(ActionNeurons)
        arr_sensory = [x.value for x in list(SensoryNeurons)]
        arr_action = [x.value for x in list(ActionNeurons)]
        arr_int_neurons = [
            x+len_sensory for x in np.arange(self.num_int_neuron)]

        # Init neuron states
        self.action_neuron_state = dict.fromkeys(arr_action, 0)
        self.int_neuron_state = dict.fromkeys(arr_int_neurons, 0)

        # General info
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
        freq = np.interp(freq, [-4, 4], [1, 5])
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
        # [] something is wrong with values

    def x_loc(self):
        x_loc = self.creature.X
        return self.map_0_1(x_loc, self.x_loc_max)

    def y_loc(self):
        y_loc = self.creature.Y
        return self.map_0_1(y_loc, self.y_loc_max)

    def dst_north(self):
        dst_north = self.creature.env.Y - self.creature.Y
        return self.map_0_1(dst_north, self.y_loc_max)

    def dst_south(self):
        dst_south = self.creature.Y
        return self.map_0_1(dst_south, self.y_loc_max)

    def dst_west(self):
        dst_west = self.creature.env.Y - self.creature.X
        return self.map_0_1(dst_west, self.x_loc_max)

    def dst_east(self):
        dst_east = self.creature.X
        return self.map_0_1(dst_east, self.x_loc_max)

    def density_around(self):
        grid_size = 5 * 5
        # Creating the pixel locations in 5 pixels around creature
        density_loc_arr = generate_pixels_around(5, self.creature_loc)
        # Calculating the unique number of difference between the total grid and proximity field
        total_arr = density_loc_arr + self.pixel_arr
        density = len(list(set(total_arr))) - len(self.pixel_arr)
        density = grid_size - density - 1  #  -1 for the creature location
        return self.map_0_1(density, grid_size-1)

    def view_forward(self, number_pixel_ahead: int = 3):
        locations_ahead = []
        # Calculate pixel locations ahead by number_pixel_ahead
        for i in range(number_pixel_ahead):
            self.creature_loc = np.add(
                tuple(self.direction), self.creature_loc)
            locations_ahead.append(tuple(self.creature_loc))
        # Search array of ahead locations if there are other creatures
        total_arr = locations_ahead + self.pixel_arr
        pixels_ahead = len(self.pixel_arr)-len(list(set(total_arr)))
        return self.map_0_1(pixels_ahead, number_pixel_ahead)

    def pheromones_around(self):
        # [] need to implement get_pheromons_around
        return 0

    def sex(self):
        # [] need to implement sex sensory
        sex = self.creature.sex
        return 0

    def age(self):
        age = self.creature.age
        return self.map_0_1(age, self.env_max_age)

    def oscillator(self):
        oscillator = self.creature.oscillator.get_value_at_step(
            self.creature.env.num_step)
        return self.map_0_1(oscillator, self.creature.oscillator.max_value)

    def __init__(self, creature: Creature):
        self.creature = creature
        self.y_loc_max = self.creature.env.Y
        self.x_loc_max = self.creature.env.X
        self.creature_loc = (self.creature.X, self.creature.Y)
        self.pixel_arr = self.creature.env.occupied_pixels
        self.direction = self.creature.last_dir.value
        self.env_max_age = self.creature.env.num_round


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

# 1. move_fr - move forward (previous direction)
# 2. move_rn - move random
# 3. move_lr - move left/right
# 4. move_ew - move east/west
# 5. move_ns - move north/south
# 6. set_osc - set oscillator period
# 7. emit_pheromone - emit pheromone


class Action:
    def __init__(self, creature: Creature):
        self.creature = creature
        self.loc = [self.creature.X, self.creature.Y]
        self.last_dir = self.creature.last_dir
        self.dir_values = [e.value for e in Directions]
        self.dir_keys = [e.name for e in Directions]
        self.max_loc = self.creature.env.X - 1

    def update_last_direction(self, last_loc: tuple, new_loc: tuple):
        last_dir = tuple([x-last_loc[idx] for idx, x in enumerate(new_loc)])
        try:
            idx = self.dir_values.index(last_dir)
            self.last_dir = Directions[self.dir_keys[idx]]
            self.creature.last_dir = self.last_dir
        except:
            return

    def move(self, direction: Directions, value: float = 1):
        new_loc = [(x + (direction.value[idx]) * value)
                   for idx, x in enumerate(self.loc)]
        # Round pixel values
        new_loc = [round(x) for x in new_loc]
        # Check so neither coordinates cannot go below zero, else set to zero
        new_loc = [0 if x < 0 else x for x in new_loc]
        # Check so neither coordinates cannot go above max, else set to max
        new_loc = [self.max_loc if x > self.max_loc else x for x in new_loc]
        # Update direction according to last loc change
        self.update_last_direction(last_loc=self.loc, new_loc=new_loc)
        self.loc = new_loc
        return new_loc

    def move_fr(self, value: float):
        direction = self.last_dir if value > 0 else OppositeDirections[self.last_dir.name].value
        new_loc = self.move(direction=direction, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def move_lr(self, value: float):
        direction = RightDirections[self.last_dir.name].value if value > 0 else LeftDirections[self.last_dir.name].value
        new_loc = self.move(direction=direction, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def move_ew(self, value: float):
        direction = Directions.EAST if value > 0 else Directions.WEST
        new_loc = self.move(direction=direction, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def move_ns(self, value: float):
        direction = Directions.NORTH if value > 0 else Directions.SOUTH
        new_loc = self.move(direction=direction, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def move_rn(self, value: float):
        rand_dir = Directions[np.random.choice(Directions._member_names_)]
        new_loc = self.move(rand_dir, value=value)
        [self.creature.X, self.creature.Y] = new_loc

    def set_osc(self, value: float):
        self.creature.oscillator.set_period(value)

    def emit_pheromone(self, value: float):
        # [] need to implement emit_pheromones
        pass


class Genome:
    def get_gene_hash(self, neuron):
        hash = hashlib.sha1(neuron).hexdigest()
        return str(hash)[:6]

    def update_input(self, inputs: np.array):
        self.inputs = inputs

    def execute_neuron_states(self):
        # Execute for all action neurons
        for h in i.action_neuron_state.items():
            if h[1]:
                getattr(action, ActionNeurons(h[0]).name)(h[1])
                log.debug(i.last_dir, i.X, i.Y)
            # [] need to multiply by synapse weights also
        i.action_neuron_state = dict.fromkeys(arr_action, 0)
        i.int_neuron_state = dict.fromkeys(arr_int_neurons, 0)
        log.debug(i.X, i.Y)

    def calculate_outputs_neurons(self):
        # Calculate output neurons =tanh(sum(input)) = -1..1 for action and internals
        for f in self.creature.int_neuron_state.items():
            inputs = np.array(self.creature.int_neuron_state[f[0]])
            self.creature.int_neuron_state[f[0]] = np.tanh(np.sum(inputs))

        for f in self.creature.action_neuron_state.items():
            inputs = np.array(self.creature.action_neuron_state[f[0]])
            self.creature.action_neuron_state[f[0]] = np.tanh(np.sum(inputs))

        log.debug('Iteration over, executing action neurons')
        log.debug(i.action_neuron_state)
        log.debug(i.int_neuron_state)

    def calculate_synapses(self):
        log.debug(self.creature.last_dir, self.creature.X, self.creature.Y)
        # [ 1., 12., 0., 1.,  0.09899215]
        # [ 0., 6., 1., 14., -2.21110532]
        # Sensory neurons output 0..1
        # Action neurons input tanh(sum(inputs)) -1..1
        # Action neurons output -4..4
        # Internal neurons input tanh(sum(inputs)) -1..1
        # Connection weights -5..5
        # If input source is action
        for gene in self.genome:
            if gene[1] in self.arr_sensory:
                input_val = getattr(
                    self.sensory, SensoryNeurons(gene[1]).name)()
            # If input source is internal neuron
            if gene[1] in self.arr_int_neurons:
                input_val = self.creature.int_neuron_state[gene[1]]

            log.debug(gene, self.creature.X, self.creature.Y, input_val)
            log.debug(SensoryNeurons(gene[1]).name if gene[1] in self.arr_sensory else gene[1], ActionNeurons(
                gene[3]).name if gene[3] in self.arr_action else gene[3])

            # If output destination is action neuron
            if gene[3] in self.arr_action:
                # getattr(action, ActionNeurons(gene(3)).name)(input_val)
                self.creature.action_neuron_state[gene[3]
                                                  ] = self.creature.action_neuron_state[gene[3]] + input_val
            # If output destination is internal neuron
            if gene[3] in self.arr_int_neurons:
                self.creature.int_neuron_state[gene[3]
                                               ] = self.creature.int_neuron_state[gene[3]] + input_val

            log.debug(self.creature.action_neuron_state)
            log.debug(self.creature.int_neuron_state)

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
        # convert to np array
        gene_array = np.array(gene_array)
        # Sort based on first gene
        gene_array = gene_array[gene_array[:, 0].argsort()]
        return gene_array

    def __init__(self, gene_size: int, num_int_neuron: int, creature: Creature):
        self.creature = creature
        self.num_int_neuron = num_int_neuron
        self.gene_size = gene_size

        self.gene_hash = []
        self.genome = self.generate_full_genome(
            gene_size=gene_size, num_int_neuron=num_int_neuron)

        # Calculate required info
        self.sensory = Sensory(creature=self.creature)
        self.action = Action(creature=self.creature)

        self.len_sensory = len(SensoryNeurons)
        self.len_action = len(ActionNeurons)

        self.arr_sensory = [x.value for x in list(SensoryNeurons)]
        self.arr_action = [x.value for x in list(ActionNeurons)]
        self.arr_int_neurons = [
            x + self.len_sensory for x in np.arange(self.num_int_neuron)]


class SensoryNeurons(Enum):
    x_loc = 0
    y_loc = 1
    dst_north = 2
    dst_south = 3
    dst_west = 4
    dst_east = 5
    density_around = 6
    view_forward = 7
    pheromones_around = 8
    sex = 9
    age = 10
    oscillator = 11


class ActionNeurons(Enum):
    move_fr = 0
    move_rn = 1
    move_lr = 2
    move_ew = 3
    move_ns = 4
    set_osc = 5
    emit_pheromone = 6
