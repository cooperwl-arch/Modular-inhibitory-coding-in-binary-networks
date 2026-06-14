# 2024/5/29
# original one neuron model
# but implement deficiency in the model
# no temperature
# This is the one running again for 1

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import random
import math

start = 1
MAX_NUMBER_CONFIGURATIONS = 1
MAX_NUMBER_TEMPERATURES1 = 1
MAX_NUMBER_TEMPERATURES = 1
Temperatures = np.array([0, 0.025, 0.05, 0.075, 0.1])
Temperatures1 = np.array([0, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16])
RatiosOfConvergence = np.zeros((MAX_NUMBER_TEMPERATURES1, MAX_NUMBER_CONFIGURATIONS), dtype=float)
weight_of_interconnections = 0.8
NUMBER_OF_NEURONS = 50
colors = [[40, 120, 181], [154, 201, 219], [248, 172, 140], [200, 36, 35], [255, 136, 132], [150, 195, 125]]
normalized_colors = np.array(colors) / 255

X1 = np.zeros((MAX_NUMBER_TEMPERATURES1, int((MAX_NUMBER_CONFIGURATIONS - start + 1) / 50)), dtype=float)

cmp_list = np.zeros((MAX_NUMBER_CONFIGURATIONS, 1000))
Legends = []


def PickNeron(number):
    return random.choice(np.arange(0, number, 1))

def save_capacity_metadata(raw_data_folder):
    metadata = {
        "data_shape": "(3, N)",
        "axis_0": "repeat",
        "axis_1": "number of patterns",
        "row_values": [
            "repeat_1",
            "repeat_2",
            "repeat_3"
        ],
        "column_values": "number of patterns tested (see csv column headers)",
        "separate_csv_files": [
            "raw_data_def_i_e_0.csv",
            "raw_data_def_i_e_25.csv",
            "raw_data_def_i_e_50.csv",
            "raw_data_def_i_e_75.csv"
        ],
        "parameter_varied_between_files": {
            "fraction of removed I-E connections (%)": [0, 25, 50, 75]
        },
        "value": "recall fraction"
    }

    with open(
            os.path.join(raw_data_folder, "metadata.json"),
            "w"
    ) as f:
        json.dump(metadata, f, indent=4)


def save_capacity_raw_data(res, x, def_i_e, folder_name):
    raw_data_folder = os.path.join(folder_name, "raw_data")
    os.makedirs(raw_data_folder, exist_ok=True)

    raw_data = np.array([res[0], res[1], res[2]])

    df = pd.DataFrame(
        raw_data,
        index=["repeat_1", "repeat_2", "repeat_3"],
        columns=x
    )

    df.to_csv(
        os.path.join(
            raw_data_folder,
            f"raw_data_def_i_e_{def_i_e}.csv"
        )
    )

def PickCon(number1, number2, deletedCon):
    list2 = [[x, y] for y in range(number2) for x in range(number1)]
    list1_set = set(map(tuple, deletedCon))
    list2 = [list(sublist) for sublist in set(map(tuple, list2)) - list1_set]
    return random.choice(list2)


def multicores(args):
    NUMBER_OF_CONFIGURATIONS, deleteRate_EtoI, deleteRate_EtoI2, deleteRate_EtoE, deleteRate_ItoE, Temperature = args
    i_e_connections_pools = []

    Configurations = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    # first step: initialize the configurations
    excitatory_to_excitatory_connections = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_NEURONS), dtype=float)

    inhibitory_to_excitatory_connections = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    excitatory_to_inhibitory_connections = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_CONFIGURATIONS), dtype=float)

    # second step: create two layers of neurons and implement learning
    # remember there are two different layers
    excitatory_neurons = np.zeros(NUMBER_OF_NEURONS, dtype=float)
    # The number of inhibitory neurons is the number of configuration
    inhibitory_neurons = np.zeros(MAX_NUMBER_CONFIGURATIONS, dtype=float)

    # we initialize the configuration
    for k in range(0, NUMBER_OF_CONFIGURATIONS):
        for i in range(0, NUMBER_OF_NEURONS):
            # not sure about this part
            Configurations[k][i] = random.choice([0, 1])

    for i in range(0, NUMBER_OF_NEURONS):
        for j in range(0, NUMBER_OF_NEURONS):
            weight = 0
            for k in range(0, NUMBER_OF_CONFIGURATIONS):
                # connections from i to j is the same as j to i
                weight += Configurations[k][i] * Configurations[k][j]

            # not quite sure this part is correct
            weight = weight / (NUMBER_OF_NEURONS * NUMBER_OF_CONFIGURATIONS)

            # not quite sure this part is correct
            excitatory_to_excitatory_connections[i][j] = weight

        for k in range(0, NUMBER_OF_CONFIGURATIONS):

            if Configurations[k][i] == 1:
                # not quite sure this part is correct
                excitatory_to_inhibitory_connections[i][k] = 1 / (NUMBER_OF_NEURONS)
            else:
                inhibitory_to_excitatory_connections[k][i] = -1
                i_e_connections_pools.append([k, i])

    deleted_Cons = random.sample(i_e_connections_pools,
                                 round((deleteRate_ItoE / 100) * len(i_e_connections_pools)))

    for a in range(0, round((deleteRate_ItoE / 100) * len(i_e_connections_pools))):
        index1 = (deleted_Cons[a])[0]
        index2 = (deleted_Cons[a])[1]
        inhibitory_to_excitatory_connections[index1][index2] = 0

    # initialize the excitatory neurons
    NumberOfConvergences = 0

    # we only randomly pick 20 configurations out of all configurations

    if NUMBER_OF_CONFIGURATIONS <= 20:
        NUMBER_OF_LOOPS = NUMBER_OF_CONFIGURATIONS
    else:
        NUMBER_OF_LOOPS = 20

    for i in range(0, NUMBER_OF_LOOPS):
        # 1000 steps
        # initialize neurons every time!

        # randomly pick a configuration
        if NUMBER_OF_CONFIGURATIONS > 20:
            random_index = PickNeron(NUMBER_OF_CONFIGURATIONS - 1)
        else:
            random_index = i

        for index in range(0, NUMBER_OF_NEURONS):
            if Configurations[random_index][index] == 1:
                excitatory_neurons[index] = 1
            else:
                excitatory_neurons[index] = 0

        # change the values of 10 percent of neurons

        reverse_neurons = random.sample(range(0, NUMBER_OF_NEURONS), round(0.1 * NUMBER_OF_NEURONS))
        for index in range(0, round(0.1 * NUMBER_OF_NEURONS)):
            key = reverse_neurons[index]
            if excitatory_neurons[key] == 1:
                excitatory_neurons[key] = 0

            elif excitatory_neurons[key] == 0:
                excitatory_neurons[key] = 1

        # we initialize our inhibitory neurons here
        inhibitory_neurons[inhibitory_neurons != 0] = 0

        sum_list = []
        # sum_list2 = []

        for a in range(0, NUMBER_OF_CONFIGURATIONS):
            sum = 0

            # not quite sure about this part. Same weight??
            # print(excitatory_to_inhibitory_connections)
            for j in range(0, NUMBER_OF_NEURONS):
                if excitatory_to_inhibitory_connections[j][a] != 0 and excitatory_neurons[j] == 1:
                    sum += 1

            sum_list.append(sum)

        # pick the inhibitory neuron with the biggest field
        inhibitory_neurons[sum_list.index(max(sum_list))] = 1

        # run the dynamics
        for m in range(0, 1000):
            tag = PickNeron(NUMBER_OF_NEURONS)
            sums = 0

            for u in range(0, NUMBER_OF_NEURONS):
                sums += excitatory_to_excitatory_connections[u][tag] * excitatory_neurons[u]

            sums += (sums) / NUMBER_OF_NEURONS

            for u in range(0, NUMBER_OF_CONFIGURATIONS):
                sums += inhibitory_to_excitatory_connections[u][tag] * inhibitory_neurons[u]
                # sums += inhibitory_to_excitatory_connections2[u][tag] * inhibitory_neurons[u]

            # implement temperature here
            if Temperature != 0:
                # remember to implement temperature
                Beta = 1 / Temperature
                Pro = 1 / (1 + math.exp(-2 * Beta * abs(sums)))
                flag1 = np.random.choice(a=[1, -1], p=[Pro, 1 - Pro])
            else:
                flag1 = 1

            # update the excitatory neurons first, then update inhibitory neurons
            if flag1 * sums >= 0:
                excitatory_neurons[tag] = 1

            elif flag1 * sums < 0:
                excitatory_neurons[tag] = 0

        # --------updating neuron part above---------

        # forth step: check the overlaps
        overlaps = 0
        for k in range(0, NUMBER_OF_NEURONS):
            # not sure about this part
            if excitatory_neurons[k] == 1 and Configurations[random_index][k] == 1:
                overlaps += 1
            elif excitatory_neurons[k] == 1 and Configurations[random_index][k] == 0:
                overlaps += (-1)
            elif excitatory_neurons[k] == 0 and Configurations[random_index][k] == 1:
                overlaps += (-1)
            else:
                overlaps += 1

        overlaps = overlaps / NUMBER_OF_NEURONS
        # 500
        if abs(overlaps) >= 0.9:
            NumberOfConvergences += 1

    print(round(NumberOfConvergences / (NUMBER_OF_LOOPS), 4))
    return round(NumberOfConvergences / (NUMBER_OF_LOOPS), 4)


######################################################################
capacities = []
folder_name = "50 I-E deficiency/inhibi_1"
if __name__ == '__main__':
    for NUMBER_OF_TEMPERATURE1 in range(0, MAX_NUMBER_TEMPERATURES1):
        Legends1 = []
        result = []
        lower_value = []
        t = 0
        for NUMBER_OF_TEMPERATURE in range(0, MAX_NUMBER_TEMPERATURES):
            #
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            tag = 0
            for def_i_e in [0, 25, 50, 75]:
                res = []
                # for NUMBER_F_CONFIGURATIONS in range(100, MAX_NUMBER_CONFIGURATIONS + 1):
                # multi process part
                for index_average in range(3):
                    x = np.arange(start, MAX_NUMBER_CONFIGURATIONS + 1, 50)
                    y0 = [0] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    y05 = [0] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    y1 = [0] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    y2 = [def_i_e] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    y3 = [Temperatures1[NUMBER_OF_TEMPERATURE1]] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    x_y = zip(x, y0, y05, y1, y2, y3)

                    pool = mp.Pool()
                    res.append(pool.map(multicores, list(x_y)))

                X1[NUMBER_OF_TEMPERATURE1] = np.arange(start, (MAX_NUMBER_CONFIGURATIONS + 1),
                                                       50) / NUMBER_OF_NEURONS


                average_res = np.mean([res[0], res[1], res[2]], axis=0)
                std_res = np.std([res[0], res[1], res[2]], axis=0)

                x = np.arange(start, MAX_NUMBER_CONFIGURATIONS + 1, 50)
                save_capacity_raw_data(res, x, def_i_e, folder_name)

                plt.errorbar(X1[NUMBER_OF_TEMPERATURE1], average_res, yerr=std_res, fmt='-o',
                             label="Reduction ratio=" + str(def_i_e),
                             color=(normalized_colors[tag][0], normalized_colors[tag][1], normalized_colors[tag][2]))
                Legends1.append(
                    "Reduction ratio=" + str(def_i_e))

                # lower_value.append(average_res[average_res < 0.4][0])

                tag = tag + 1

    plt.legend()
    file_name = folder_name + "/capacities"

    if not os.path.exists(file_name):
        os.makedirs(file_name)

    # save the json file
    save_capacity_metadata(folder_name)


    plt.xlabel(r'Number of configurations / $n_e$')
    plt.ylabel("Recall Fraction")
    plt.ylim(-0.2, 1.2)

    plt.savefig(file_name + "/capacity_50_ex-neurons_system_1_in_neuron_I-E_deficiency" + ".pdf", format='pdf')
    plt.savefig(file_name + "/capacity_50_ex-neurons_system_1_in_neuron_I-E_deficiency" + ".svg", format='svg')
    plt.show()