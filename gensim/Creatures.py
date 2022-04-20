import sys
import logging
import uuid
import hashlib
import numpy as np
from enum import Enum

from gensim.Neurons import Neuron

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Creature:
    def mutate(self):
        pass

    def generate_genes():
        pass

    def generate_synapse():
        pass

    def get_gene_hash(self):
        hash = hashlib.sha1(self.neuron_array).hexdigest()
        return str(hash)[:6]

    def __init__(self, gene_size: int, num_int_neuron: int):
        # Generate gene pool for Creature
        neuron_array = np.array([0, 0, 0, 0, 0])
        gene_array = []
        for i in range(gene_size):
            gene = Gene(neuron_type=GeneType.NORMAL)
            neuron_array = np.vstack((neuron_array, gene.neuron))
            gene_array.append(gene)
        for i in range(num_int_neuron):
            gene = Gene(neuron_type=GeneType.INTERNAL,
                        num_int_neuron=num_int_neuron)
            neuron_array = np.vstack((neuron_array, gene.neuron))
            gene_array.append(gene)
        # Saving arrays
        neuron_array = np.flip(neuron_array, axis=0)
        self.neuron_array = neuron_array[1:]  # Slicing off first empty neuron
        self.gene_array = gene_array
        self.id = uuid.uuid4()
        log.info(
            f"Creature {str(self.id)[:8]} neuron array:\n{self.neuron_array}")
        # Location data
        self.X = 0
        self.Y = 0
        self.last_dir = Directions.NULL
        # Setting oscillator with a random init frequency
        self.oscillator = Oscillator(np.random.uniform(1, 5))


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


class Directions(Enum):
    EAST = [1, 0]
    WEST = [-1, 0]
    NORTH = [0, 1]
    SOUTH = [0, -1]
    NULL = [0, 0]


class Action:
    def __init__(self, creature: Creature):
        self.creature = creature

    def move(self, direction):
        loc = [self.creature.X, self.creature.Y]
        new_loc = [x + Directions[direction].value[idx]
                   for idx, x in enumerate(loc)]
        # Check so neither coordinates cannot go below zero, else set to zero
        new_loc = [0 if x < 0 else x for x in new_loc]
        [self.creature.X, self.creature.Y] = new_loc

    def set_oscillator(self, period):
        pass

    def emit(self):
        pass


class GeneType(Enum):
    INTERNAL = 1
    NORMAL = 0


class Gene:
    def update_input(self, inputs: np.array):
        self.inputs = inputs

    def calculate_synapse(self):
        pass

    def generate_gene(self, neuron_type: GeneType, num_int_neuron: int):
        # [source_type][from_neuron_id][destination_type][to_neuron_id][synapse_weight]
        neuron = Neuron()

        # Calculate max lenghts
        max_sensory_len = neuron.list_sensory_size()
        max_action_len = neuron.list_action_size()
        from_neuron_len = max_sensory_len + num_int_neuron
        to_neuron_len = max_action_len + num_int_neuron

        # Creating neurons
        if neuron_type == GeneType.NORMAL:
            source_type = 0
            from_neuron_id = np.random.randint(max_sensory_len)

            destination_type = np.random.randint(2)
            to_neuron_id = np.random.randint(
                max_action_len) if destination_type == 0 else np.random.randint(max_action_len+1)

        if neuron_type == GeneType.INTERNAL:
            source_type = 1
            from_neuron_id = np.random.randint(
                max_sensory_len+1, from_neuron_len+1)

            destination_type = np.random.randint(2)
            to_neuron_id = np.random.randint(max_action_len) if destination_type == 0 else np.random.randint(
                max_action_len+1, to_neuron_len+1)

        # Setting synapse weight between 1 and 5
        synapse_weight = np.random.uniform(low=-5, high=5)

        array = np.array([source_type, from_neuron_id,
                         destination_type, to_neuron_id, synapse_weight])

        return array

    def __init__(self, neuron_type: GeneType, num_int_neuron=0):

        self.neuron = self.generate_gene(neuron_type, num_int_neuron)
        self.neuron_shape = self.neuron.shape[0]
        self.inputs = np.empty(self.neuron_shape, dtype=object)
