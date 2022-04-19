# GenSim
## Genetic simulation of evolution

This game would like to create a simulation enviroment for simple genetic evolution. Each creature has multiple input and output neurons for percieving its enviroment and interact within it.

Based on selection criteria at the each of the rounds a creature can survive and create an offspring with a possible mutation in its genes.

### Enviroment
- X * Y size pixels grid
- Each pixel can host a single individual
- Selection criteria should be defined based on areas of the grid 

The enviroment is made up of an X times Y **pixel grid**. Time is divided for **rounds**, which gives an opportunity for each creature to fulfill the **selection criteria**. At the start of the simulation a **population size** of n creatures is defined. After each round creatures are evaluated for the selection criteria and if they success can **create offsprings** for the next round with a **mutation** probability. 

 Creatures can move around in the enviroment based on their action neuron outputs. A pixel can only host a single creature on it, so creatures can be blocking each other.

## Creatures

Each creature is made up of neurons. Neurons can help the creature to sense and interact with the enviroment. Synapses make connections between between sensory and action neurons and optionally route through internal neurons.

Creatures are initially created with a specific number of synapses between these neurons. This greatly effects the intelligence of the creatures. 

Each creature holds the information of its genes (number of synapses) and has a different color based on this information, so the genetic difference is visible on the simulation. This information is stored in a numpy array as below:

```
[source_type][from_neuron_id][destination_type][to_neuron_id][synapse_weight]

source_type(0,1) = if the input comes from a sensory or internal neuron

from_neuron_id = the unique id of the source neuron

destination_type = if the output goes a sensory or internal neuron

to_neuron_id = the unique id of the destination neuron

synapse_weight = the weight of the connection
```

### Additional information on neuron math

- Sensory neurons emit a value between `0..1`
- Internal neurons are outputting `=tanh(sum(input)) = -1..1`
- Action neurons are acting upont `=tanh(sum(input)) = -1..1`
- Connections has a weight between `-5..5

Each sensory neuron outputs a float number (-1,+1) 

### Type of neurons to consider
- sensory of internal or enviromental state
- internal neuron
- action neuron to interact in the enviroment

### Sensory Neurons
- Sx - location on the X axis
- Sy - location on the Y axis
- Dn - distance from north
- Ds - distance from south
- Dw - distance from west
- De - distance from east
- De - density around: how many individuals are around (8 pixels)
- Va - view ahead forward: is there an individual in the next 3 pixels
- Ph - pheromons detected around (5x5 pixels)
- Se - sex
- Ag - age
- Os - internal oscillator signal

### Action Neurons
- Mf - move forward (previous direction)
- Mrv - move reverse/backwards
- Mrn - move random
- Mlr - move left/right
- Mew - move east/west
- Mns - move north/south
- So - set oscillator period
- Ep - emit pheromone