import sys
import os
import logging
import uuid
import hashlib
import graphviz
import numpy as np
from enum import Enum

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class Creature:
    def mutate(self):
        pass

    def generate_genes():
        pass

    def generate_synapse():
        pass

    def create_graph_img(self):
        genome_hash = self.get_genome_hash()
        dot = graphviz.Digraph(comment=genome_hash)
        for idx, i in enumerate(self.neuron_array):
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
        save_path = f"{self.sim_id}/{genome_hash}"
        dot.render(save_path, view=True, format="png")
        # Remove dotfile
        os.remove(save_path)

    def get_genome_hash(self):
        hash = hashlib.sha1(self.neuron_array).hexdigest()
        return str(hash)[:6]

    def __init__(self, simulation_id: str, gene_size: int, num_int_neuron: int):
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
        self.neuron_array = neuron_array[1:]  # Slicing off first empty neuron
        self.gene_array = gene_array
        self.id = uuid.uuid4()
        self.sim_id = simulation_id
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
    def get_gene_hash(self):
        hash = hashlib.sha1(self.neuron).hexdigest()
        return str(hash)[:6]

    def update_input(self, inputs: np.array):
        self.inputs = inputs

    def calculate_synapse(self):
        pass

    def generate_gene(self, neuron_type: GeneType, num_int_neuron: int):
        # [source_type][from_neuron_id][destination_type][to_neuron_id][synapse_weight]

        # Calculate max lenghts
        max_sensory_len = len(SensoryNeurons)
        max_action_len = len(ActionNeurons)
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
        self.gene_hash = self.get_gene_hash()


class SensoryNeurons(Enum):
    Sx = 1
    Sy = 2
    Dn = 3
    Ds = 4
    Dw = 5
    De = 6
    Da = 7
    Va = 8
    Ph = 9
    Se = 10
    Ag = 11
    Os = 12


class ActionNeurons(Enum):
    Mf = 1
    Mrv = 2
    Mrn = 3
    Mlr = 4
    Mew = 5
    Mns = 6
    So = 7
    Ep = 8
