# Modular inhibitory-excitatory binary network under connection deficiency.
This repository contains the Python code for the Figure 4 and Figure 5 in the paper studying the impacts of connection loss on the memory storage.

## Overview 
The network consists of:
  1. A population of excitatory neurons storing binary memory patterns.
  2. A population of inhibitory neurons organized by modules acting as pattern-selective competition units.
  3. Asychronous update dynamics
  4. Random perturbation of initial states to 10% of neurons before recall.

For each level of connection loss, the simulation measures the fraction of successfully recalled memories. 

## Dependencies 
os,
matplotlib,
numpy,
multiprocessing,
random,
pandas,
json

## Figure 5

### Parameters:
Default setting:
| Parameter                    |              Value |
| ---------------------------- | -----------------: |
| Number of excitatory neurons |                 50 |
| Number of stored patterns    |                150 |
| Initial perturbation         | 10% flipped neurons|
| Dynamics length              |  1000 update steps |
| Number of repetitions        |                  3 |

Connection deletion fractions:
### E->E connections
0%, 25%, 50%, 75%, 90%

### I->E connections
0%, 10%, 15%, 25%, 50%, 75%

### Output
The code generates:

#### Heatmap of recall performance
results/deficiencies_matrix/deficiency_matrix.png

showing:
 1. mean recall fraction
 2. standard deviation across repetitions

#### Raw data
results/results_repeat_{i}.csv

(i=1, 2, 3)

#### A json file labeling the information of the raw data

### Running
Python Figure 5.py

The simulation uses Python multiprocessing for parallel execution.


## Figure 4
A, B, C after the name of every file "Figure_4"are the labels of the columns of the panels, 
representing E-E, E-I, I-E connections.

### Parameters:
Default setting:
| Parameter                    |              Value |
| ---------------------------- | -----------------: |
| Number of excitatory neurons |                 50 |
| Number of inhibitory neurons in every module    |           1, 3, or 5|
| Number of stored patterns    |           1 to 1500|
| Step length of iterating over the number of stored patterns|      50|
| Initial perturbation         | 10% flipped neurons|
| Dynamics length              |  1000 update steps |
| Number of repetitions        |                  3 |

Connection deletion fractions:
### E->E connections
0%, 25%, 50%, 75%

### Output
The code generates:

#### Capacity curves
For each reduction rate of e-e connections:

/capacity_50_ex-neurons_system_{i}_in_neuron_E-E_deficiency.pdf

/capacity_50_ex-neurons_system_{i}_in_neuron_E-E_deficiency.svg

(i=1, 3, 5)

showing:
 1. mean recall fraction for different numbers of patterns
 2. standard deviation across repetitions

#### Raw data
50 E-E deficiency/inhibi_{i}/raw_data/raw_data_def_e_e_{def_e_e}.csv (Figure 4A)
50 E-I deficiency/inhibi_{i}/raw_data/raw_data_def_e_e_{def_e_e}.csv (Figure 4B)
50 I-E deficiency/inhibi_{i}/raw_data/raw_data_def_e_e_{def_e_e}.csv (Figure 4C)

(def_e_e=0, 25, 50, 75)

(i=1, 3, 5)

#### A json file labeling the information of the raw data

### Running

Python Figure 4_i_inhibi_j.py

(i=A, B, C. i is the label of the columns of the panels, representing E-E, E-I, I-E connections.
j=1, 3, 5).

The simulation uses Python multiprocessing for parallel execution.








