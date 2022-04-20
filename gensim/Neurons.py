import logging

log = logging.getLogger(__name__)


class Neuron:
    def list_neurons(self):
        print(self.sensory_neuron_types)
        print(self.action_neuron_types)

    def list_sensory_size(self):
        return len(self.sensory_neuron_types)

    def list_action_size(self):
        return len(self.action_neuron_types)

    def __init__(self):
        self.sensory_neuron_types = {
            'Sx':   0,
            'Sy':   0,
            'Dn':   0,
            'Ds':   0,
            'Dw':   0,
            'De':   0,
            'De':   0,
            'Va':   0,
            'Ph':   0,
            'Se':   0,
            'Ag':   0,
            'Os':   0,
        }
        self.action_neuron_types = {
            'Mf':   0,
            'Mrv':  0,
            'Mrn':  0,
            'Mlr':  0,
            'Mew':  0,
            'Mns':  0,
            'So':   0,
            'Ep':   0,
        }
