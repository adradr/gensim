import os
import logging
import uuid
import hashlib
import graphviz
import numpy as np
from enum import Enum
from PyProbs import Probability as pr

log = logging.getLogger('gensim')


class Creature:
    # [] implement graph img execution when new offspring is created
    def create_graph_img(self, view_img=False):
        genome_hash = self.get_genome_hash()
        save_path = f"{self.env.sim_gendir}{self.id_short}"
        dot = graphviz.Digraph(comment=genome_hash)
        dot.attr(ratio='auto', size='30',
                 fontname="Helvetica", layout="neato", overlap="prism",
                 label=f"Creature id: {self.id_short}\nGenome hash: {genome_hash}\n{self.sim_id}")
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
            rounded_weight = np.round(i[4], 2)
            edge_color = "#30b854" if rounded_weight > 0 else "#b01726"
            edge_width = str(abs(rounded_weight)+1)
            dot.edge(a, b, '', {
                     "penwidth": edge_width, "color": edge_color})
            # dot.unflatten(stagger=5)
        dot.render(save_path, view=view_img, format="png")
        # Remove dotfile
        os.remove(save_path)

    def get_genome_hash(self):
        hash = hashlib.sha1(self.gene_array).hexdigest()
        return str(hash)[:6]

    # [] combined reinit and init so there is no code duplication
    def reinit_offspring(self):
        # General info
        self.id = uuid.uuid4()
        self.id_short = str(self.id)[:8]

        # Location data
        self.X = 0
        self.Y = 0
        self.last_dir = Directions[Directions._member_names_[
            np.random.randint(len(Directions))]]

        # Setting oscillator with a random init frequency
        freq = np.random.uniform(1, 5)
        self.oscillator = Oscillator(freq=freq, max_lenght=self.env.num_steps)

        # Setting init attributes
        self.age = 0
        self.sex = np.random.randint(2)  #  0 - male, 1 - female

    def __init__(self, env, gene_size: int, num_int_neuron: int):
        # General info
        self.id = uuid.uuid4()
        self.id_short = str(self.id)[:8]
        self.env = env
        self.sim_id = env.sim_dir

        # Location data
        self.X = 0
        self.Y = 0
        self.last_dir = Directions[Directions._member_names_[
            np.random.randint(len(Directions))]]

        # Setting oscillator with a random init frequency
        freq = np.random.uniform(1, 5)
        self.oscillator = Oscillator(freq=freq, max_lenght=self.env.num_steps)

        # Setting init attributes
        self.age = 0
        self.sex = np.random.randint(2)  #  0 - male, 1 - female

        # Get genome
        self.gene_size = gene_size
        self.num_int_neuron = num_int_neuron
        self.genome = Genome(
            gene_size=gene_size, num_int_neuron=num_int_neuron, creature=self)
        self.gene_array = self.genome.genome
        log.debug(
            f"Creature {self.id_short} neuron array:\n{self.gene_array}")


class Oscillator:
    def get_value_at_step(self, step):
        return self.signal[step]

    def set_period(self, freq: float):
        freq = np.interp(freq, [-4, 4], [1, 5])
        assert 5 >= freq >= 1, "Use value between 5 and 1"
        self.freq = freq
        time = np.arange(0, self.max_lenght, 0.1)
        signal = np.sin(time / self.freq) + 1
        self.signal = signal

    def __init__(self, freq: float, max_lenght: int):
        self.freq = freq
        self.max_lenght = max_lenght
        self.set_period(freq=self.freq)
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
    def generate_pixels_around(self, grid_size: int, current_location: tuple):
        # Getting a nullpoint for the grid_size * grid_size search grid
        half_grid_size = np.ceil(grid_size/2).astype(int)
        null_point = [x-half_grid_size for x in current_location]
        # Creating the pixel locations in the pixels around creature
        density_loc_arr = []
        for i in range(1, grid_size + 1):
            for f in range(1, grid_size + 1):
                new_loc = (null_point[0] + f, null_point[1] + i)
                density_loc_arr.append(new_loc)
        return density_loc_arr

    def map_0_1(self, value: int, max_value: int):
        return value / max_value

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
        dst_west = self.creature.X
        return self.map_0_1(dst_west, self.x_loc_max)

    def dst_east(self):
        dst_east = self.creature.env.X - self.creature.X
        return self.map_0_1(dst_east, self.x_loc_max)

    def density_around(self):
        grid_size = 5 * 5
        # Creating the pixel locations in 5 pixels around creature
        density_loc_arr = self.generate_pixels_around(5, self.creature_loc)
        # Calculating the unique number of difference between the total grid and proximity field
        total_arr = density_loc_arr + self.creature.env.occupied_pixels
        density = len(list(set(total_arr))) - \
            len(self.creature.env.occupied_pixels)
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
        total_arr = locations_ahead + self.creature.env.occupied_pixels
        pixels_ahead = len(
            self.creature.env.occupied_pixels)-len(list(set(total_arr)))
        return self.map_0_1(pixels_ahead, number_pixel_ahead)

    def pheromones_around(self):
        # [] need to implement get_pheromons_around
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
        # self.loc = [self.creature.X, self.creature.Y]
        self.last_dir = self.creature.last_dir
        self.dir_values = [e.value for e in Directions]
        self.dir_keys = [e.name for e in Directions]
        self.max_loc = self.creature.env.X - 1

    def update_loc(self):
        self.loc = [self.creature.X, self.creature.Y]

    def update_cr_loc(self, new_loc):
        [self.creature.X, self.creature.Y] = new_loc

    def update_last_direction(self, last_loc: tuple, new_loc: tuple):
        last_dir = tuple([x-last_loc[idx] for idx, x in enumerate(new_loc)])
        try:
            idx = self.dir_values.index(last_dir)
            self.last_dir = Directions[self.dir_keys[idx]]
            self.creature.last_dir = self.last_dir
        except:
            return

    # [x] need to forbid to have multiple creatures move to the same pixel

    def move(self, direction: Directions, value: float = 1):
        # Update current creature location
        self.loc = [self.creature.X, self.creature.Y]

        # Define probability and return current location if its false and exit
        # Input values are -1...1. We take its abs value and calculate a probability
        # Action neuron only fires if the output value's probability turns to true
        value = np.round(np.abs(value), 4)
        if not pr.Prob(value):
            return self.loc

        # Get current occupied state
        occupied_pixels = self.creature.env.occupied_pixels
        try:
            occupied_pixels.remove(tuple(self.loc))
        except:
            pass

        # Calculate new location
        new_loc = [(x + (direction.value[idx]) * value)
                   for idx, x in enumerate(self.loc)]
        # Round pixel values
        new_loc = [np.round(x).astype(int) for x in new_loc]
        # Check so neither coordinates cannot go below zero, else set to zero
        new_loc = [0 if x < 0 else x for x in new_loc]
        # Check so neither coordinates cannot go above max, else set to max
        new_loc = [self.max_loc if x > self.max_loc else x for x in new_loc]
        # Check if the pixel is not occupied yet, else reset location to current
        log.debug(f"Loc - {self.creature.id_short} {self.loc, new_loc}")
        if tuple(new_loc) in occupied_pixels:
            log.debug(f"Loc collision! - {self.loc, new_loc}")
            new_loc = self.loc

        # Update direction according to last loc change
        self.update_last_direction(last_loc=self.loc, new_loc=new_loc)
        self.loc = new_loc
        # Calculate new occupied pixels
        self.creature.env.occupied_pixels = self.creature.env.calc_occupied_pixels()

        return new_loc

    def move_fr(self, value: float):
        direction = self.last_dir if value > 0 else OppositeDirections[self.last_dir.name].value
        new_loc = self.move(direction=direction, value=value)
        self.update_cr_loc(new_loc=new_loc)

    def move_lr(self, value: float):
        direction = RightDirections[self.last_dir.name].value if value > 0 else LeftDirections[self.last_dir.name].value
        new_loc = self.move(direction=direction, value=value)
        self.update_cr_loc(new_loc=new_loc)

    def move_ew(self, value: float):
        direction = Directions.EAST if value > 0 else Directions.WEST
        new_loc = self.move(direction=direction, value=value)
        self.update_cr_loc(new_loc=new_loc)

    def move_ns(self, value: float):
        direction = Directions.NORTH if value > 0 else Directions.SOUTH
        new_loc = self.move(direction=direction, value=value)
        self.update_cr_loc(new_loc=new_loc)

    def move_rn(self, value: float):
        rand_dir = Directions[np.random.choice(Directions._member_names_)]
        new_loc = self.move(rand_dir, value=value)
        self.update_cr_loc(new_loc=new_loc)

    def set_osc(self, value: float):
        self.creature.oscillator.set_period(value)

    def emit_pheromone(self, value: float):
        # [] need to implement emit_pheromones
        pass

# [x] debug why are they moving north/east mostly?
# [x] need to multiply by synapse weights also
# [] create a better genome coloring method so similars are close in color


def log_states_reset(creature):
    log.debug(f"{creature.id_short} - Resetting neuron states...")
    log.debug(
        f"{creature.id_short} - Neuron state (int): {creature.genome.int_neuron_state}")
    log.debug(
        f"{creature.id_short} - Neuron state (act): {creature.genome.action_neuron_state}")


def log_neuron_update(creature, gene, input_val):
    log.debug(
        f"{creature.id_short} - Neuron updated: input:{'N'+str(gene[1]) if gene[0] == 1 else SensoryNeurons(gene[1]).name} / output:{'N'+str(gene[3]) if gene[2] == 1 else ActionNeurons(gene[3]).name} / value: {input_val}")


class NeuronCalculator:

    # [ 1., 12., 0., 1.,  0.09899215]
    # [ 0., 6., 1., 14., -2.21110532]
    # Sensory neurons output 0..1
    # Action neurons input tanh(sum(inputs)) -1..1
    # Action neurons output -4..4
    # Internal neurons input tanh(sum(inputs)) -1..1
    # Connection weights -5..5

    # [ 0.     0.     1.    11.    -3.191]
    # [ 0.     2.     0.     2.    -3.672]
    # ...
    # [ 1.    11.     1.    11.    -2.126]

    def calc_tanh(self, inputs):
        return np.tanh(np.sum(inputs))

    def reset_neuron_states(self, creature: Creature):
        # Log states before erasing
        log_states_reset(creature)
        # Reset neurons
        creature.genome.action_neuron_state = {key: [0]
                                               for key in creature.genome.arr_action}
        creature.genome.int_neuron_state = {key: [0]
                                            for key in creature.genome.arr_int_neurons}

    def calc_neurons(self, creature: Creature):
        for gene in creature.genome.genome:
            # ----- Input -------
            if gene[0] == 0:  # If sensory neuron
                input_val = getattr(
                    creature.genome.sensory, SensoryNeurons(gene[1]).name)()
                input_val *= gene[4]  #  Multiply by weights
            if gene[0] == 1:  # If internal neuron
                # Get the internal neuron from previous step state
                try:
                    input_val = self.calc_tanh(
                        creature.genome.int_neuron_state_prev[gene[1]])
                # If not avaiable previous state get it from current
                except:
                    input_val = self.calc_tanh(
                        creature.genome.int_neuron_state[gene[1]])  # Calculate with tanh formula
                input_val *= gene[4]  #  Multiply by weights
            # ----- Output -------
            if gene[2] == 0:  #  If action neuron
                creature.genome.action_neuron_state[gene[3]].append(input_val)
            if gene[2] == 1:  #  If internal neuron
                creature.genome.int_neuron_state[gene[3]].append(input_val)

            # Logging debug
            log_neuron_update(creature, gene, input_val)

    def calc_action_outputs(self, creature: Creature):
        # Iterate over action neurons and calculate final output values
        for k, v in creature.genome.action_neuron_state.items():
            creature.genome.action_neuron_state[k] = self.calc_tanh(v)

    def execute_actions(self, creature: Creature):
        # Save previous states
        creature.genome.int_neuron_state_prev = creature.genome.int_neuron_state

        log.debug(
            f"{creature.id_short} - Executing action neurons...")
        log.debug(
            f"{creature.id_short} - Neuron state (act): {creature.genome.action_neuron_state}")

        # Iterate action neurons and execute
        for k, v in creature.genome.action_neuron_state.items():
            if v:  # If there is a value
                # Execute action
                getattr(creature.genome.action, ActionNeurons(k).name)(v)
                # Update occupied pixels after each new move
                creature.env.occupied_pixels = creature.env.calc_occupied_pixels()

##############################################################################################################


class Genome:
    def set_random_genome_from_creatures(self, creatures: list[Creature]):
        gene_size = creatures[0].gene_size
        # Get a list of genes from creatures, and reshape into a 2D array
        gene_pool = np.vstack([x.genome.genome for x in creatures])
        # Generate a random list of genes picked from a list of creatures
        gene_idx_selection = np.random.choice(
            range(len(gene_pool)), gene_size)
        # Select genes from parents' gene pool and save in self.genome
        new_genome = np.vstack([gene_pool[x] for x in gene_idx_selection])
        # Cast all values to int
        # new_genome = [[int(y) if idx < 4 else y for idx,
        #                y in enumerate(x)] for x in new_genome]
        self.genome = new_genome

    def mutate_genome(self, mutation_probability: float):
        # Going through the genome
        for idx, gene in enumerate(self.genome):
            # If probability is True
            if pr.Prob(mutation_probability):
                # Generate new gene, store it with the hash
                new_gene = self.generate_gene(self.int_neuron_arr)
                self.genome[idx] = new_gene
                self.gene_hash[idx] = self.get_gene_hash(new_gene)

    def get_gene_hash(self, neuron):
        # hash = hashlib.sha1(neuron).hexdigest()
        # return str(hash)[:6]
        for idx, i in enumerate(neuron):
            if idx == 1:
                r = i
            elif idx == 3:
                g = i
            elif idx == 4:
                b = i

        r_max = self.len_sensory + self.num_int_neuron
        g_max = self.len_action + self.num_int_neuron
        b_max = self.weight_min + self.weight_max

        r = np.interp(r, [0, r_max], [0, 256]).astype(int)
        g = np.interp(g, [0, g_max], [0, 256]).astype(int)
        b = np.interp(b, [-5, 5], [0, 256]).astype(int)
        rgb = np.array([r, g, b])

        hexes = [hex(x) for x in rgb]
        hex_color = ''.join(hexes).replace('0x', '')
        return hex_color

    def generate_gene(self, int_neuron_arr: list):
        # [source_type][from_neuron_id][destination_type][to_neuron_id][synapse_weight]

        source_type = np.random.randint(2)
        from_neuron_id = np.random.randint(
            self.len_sensory) if source_type == 0 else int(np.random.choice(int_neuron_arr, 1))
        destination_type = np.random.randint(2)
        to_neuron_id = np.random.randint(
            self.len_action) if destination_type == 0 else int(np.random.choice(int_neuron_arr, 1))

        # Setting synapse weight between 1 and 5, rounding to 3 digits
        synapse_weight = np.random.uniform(low=-5, high=5)
        synapse_weight = np.round(synapse_weight, 3)

        array = np.array([source_type, from_neuron_id,
                         destination_type, to_neuron_id, synapse_weight])

        return array

    def generate_full_genome(self, gene_size: int, num_int_neuron: int):
        # Get internal neuron list
        self.int_neuron_arr = np.arange(
            self.len_action, self.len_action + self.num_int_neuron)
        # Generate gene pool for Creature
        gene_array = []
        for i in range(gene_size):
            gene = self.generate_gene(self.int_neuron_arr)
            # Generating gene
            gene_array.append(gene)
            # Getting hash of gene and storing
            self.gene_hash.append(self.get_gene_hash(gene))
        # Convert to np array
        gene_array = np.array(gene_array)
        # Sort based on first gene
        gene_array = gene_array[gene_array[:, 0].argsort()]
        return gene_array

    def __init__(self, gene_size: int,
                 num_int_neuron: int,
                 creature: Creature,
                 weight_range: int = 5):
        self.creature = creature
        self.num_int_neuron = num_int_neuron
        self.gene_size = gene_size
        self.weight_min = -weight_range
        self.weight_max = weight_range

        # Calculate required info
        self.sensory = Sensory(creature=self.creature)
        self.action = Action(creature=self.creature)

        self.len_sensory = len(SensoryNeurons)
        self.len_action = len(ActionNeurons)

        self.arr_sensory = [x.value for x in list(SensoryNeurons)]
        self.arr_action = [x.value for x in list(ActionNeurons)]
        self.arr_int_neurons = [
            x + self.len_action for x in np.arange(self.num_int_neuron)]

        # Init neuron states
        self.action_neuron_state = {key: [0] for key in self.arr_action}
        self.int_neuron_state = {key: [0] for key in self.arr_int_neurons}

        # Generate genes
        self.gene_hash = []
        self.genome = self.generate_full_genome(
            gene_size=gene_size, num_int_neuron=num_int_neuron)


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
    age = 9
    oscillator = 10


class ActionNeurons(Enum):
    move_fr = 0
    move_rn = 1
    move_lr = 2
    move_ew = 3
    move_ns = 4
    set_osc = 5
    emit_pheromone = 6
