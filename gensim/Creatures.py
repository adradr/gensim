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
            gene = Gene(num_int_neuron=num_int_neuron)
            neuron_array = np.vstack((neuron_array, gene.neuron))
            gene_array.append(gene)
        # Saving arrays
        self.neuron_array = neuron_array[1:]
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

# [] need to define a method so creatures cannot move out of boundaries


class Action:
    def __init__(self, creature: Creature):
        self.creature = creature

    def move(self, direction):
        loc = [self.creature.X, self.creature.Y]
        new_loc = [x + Directions[direction].value[idx]
                   for idx, x in enumerate(loc)]
        [self.creature.X, self.creature.Y] = new_loc

    def set_oscillator(self, period):
        pass

    def emit(self):
        pass


class Gene:
    def generate_gene(self, from_neuron_len, to_neuron_len):
        source_type = np.random.randint(2)
        destination_type = np.random.randint(2)
        from_neuron_id = np.random.randint(from_neuron_len)
        to_neuron_id = np.random.randint(to_neuron_len)
        synapse_weight = np.random.uniform(low=-5, high=5)
        array = np.array([source_type, destination_type,
                          from_neuron_id, to_neuron_id, synapse_weight])
        return array

    def __init__(self, num_int_neuron: int):
        # [source_type][from_neuron_id][destination_type][to_neuron_id][synapse_weight]
        neuron = Neuron()
        max_sensory_len = neuron.list_sensory_size()
        max_action_len = neuron.list_action_size()
        from_neuron_len = max_sensory_len + num_int_neuron
        to_neuron_len = max_action_len + num_int_neuron

        self.neuron = self.generate_gene(from_neuron_len, to_neuron_len)
