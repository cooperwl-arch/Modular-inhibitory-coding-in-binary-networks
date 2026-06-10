# Edition 25: old model
# 2023/10/22
# A new model in which the number of the inhibitory neurons equals to the number of configurations
# Same as main20, but this print m of five neurons
# version of multiprocessing

import os
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import random
import pandas as pd
import json

start = 150
MAX_NUMBER_CONFIGURATIONS = 150
MAX_NUMBER_TEMPERATURES = 1
Temperatures = np.array([0, 0.025, 0.05, 0.075, 0.1])

MAX_NUMBER_TEMPERATURES1 = 1
Temperatures1 = np.array([0, 0.005, 0.010, 0.015, 0.020, 0.025])
distance = 5
RatiosOfConvergence = np.zeros((MAX_NUMBER_TEMPERATURES1, MAX_NUMBER_CONFIGURATIONS), dtype=float)
weight_of_interconnections = 0.8
NUMBER_OF_NEURONS = 50

X1 = np.zeros((MAX_NUMBER_TEMPERATURES1, MAX_NUMBER_CONFIGURATIONS - start + 1), dtype=float)

cmp_list = np.zeros((MAX_NUMBER_CONFIGURATIONS, 1000))
Legends = []


def PickNeron(number):
    return random.choice(np.arange(0, number, 1))


def PickCon(number1, number2, deletedCon):
    list2 = [[x, y] for y in range(number2) for x in range(number1)]

    list1_set = set(map(tuple, deletedCon))
    list2 = [list(sublist) for sublist in set(map(tuple, list2)) - list1_set]
    return random.choice(list2)

def save_raw_data(result):
    result_array = np.array(result).reshape(3, 5, 6)

    EE_delete = [0, 25, 50, 75, 90]
    IE_delete = [0, 10, 15, 25, 50, 75]

    raw_data_folder = "results/raw_data"
    os.makedirs(raw_data_folder, exist_ok=True)

    # save each repeat as a 5x6 table
    for i in range(result_array.shape[0]):
        df = pd.DataFrame(
            result_array[i],
            index=EE_delete,
            columns=IE_delete
        )

        df.to_csv(
            os.path.join(raw_data_folder, f"results_repeat_{i+1}.csv")
        )

    # save metadata
    metadata = {
        "data_shape": "(3, 5, 6)",
        "axis_0": "repeat",
        "axis_1": "fraction of removed E-E connections (%)",
        "axis_2": "fraction of removed I-E connections (%)",
        "rows_in_each_csv": EE_delete,
        "columns_in_each_csv": IE_delete,
        "value": "recall ratio"
    }

    with open(os.path.join(raw_data_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def multicores(args):
    NUMBER_OF_CONFIGURATIONS, deleteRate_EtoE, deleteRate_ItoE = args
    global ifEnd
    Configurations = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    # first step: initialize the configurations
    excitatory_to_excitatory_connections = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_NEURONS), dtype=float)
    inhibitory_to_excitatory_connections = np.zeros((MAX_NUMBER_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    excitatory_to_inhibitory_connections = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_CONFIGURATIONS), dtype=float)

    # second step: create two layers of neurons and implement learning
    # remember there are two different layers
    excitatory_neurons = np.zeros(NUMBER_OF_NEURONS, dtype=float)
    # The number of inhibitory neurons is the number of configuration
    inhibitory_neurons = np.zeros(MAX_NUMBER_CONFIGURATIONS, dtype=float)

    # we initialize the configuration
    for k in range(0, NUMBER_OF_CONFIGURATIONS):
        for i in range(0, NUMBER_OF_NEURONS):

            Configurations[k][i] = random.choice([0, 1])

    for i in range(0, NUMBER_OF_NEURONS):
        for j in range(0, NUMBER_OF_NEURONS):
            weight = 0
            for k in range(0, NUMBER_OF_CONFIGURATIONS):
                # connections from i to j is the same as j to i
                weight += Configurations[k][i] * Configurations[k][j]

            weight = weight / (NUMBER_OF_NEURONS * NUMBER_OF_CONFIGURATIONS)

            excitatory_to_excitatory_connections[i][j] = weight

        for k in range(0, NUMBER_OF_CONFIGURATIONS):

            if Configurations[k][i] == 1:
                excitatory_to_inhibitory_connections[i][k] = 1 / (NUMBER_OF_NEURONS)
            else:
                inhibitory_to_excitatory_connections[k][i] = -1

    inhibitory_to_excitatory_connections_null = [(i1, j1) for i1, row in enumerate(inhibitory_to_excitatory_connections)
                                                 for j1, value in enumerate(row) if value == 0]

    # start to delete excitatory to excitatory connections
    deletedCon = []
    for a in range(0, round((deleteRate_EtoE / 100) * 2500)):
        random_connection = PickCon(NUMBER_OF_NEURONS, NUMBER_OF_NEURONS, deletedCon)
        excitatory_to_excitatory_connections[random_connection[0]][random_connection[1]] = 0
        deletedCon.append(random_connection)

    # start to delete inhibitory to excitatory connections
    deletedCon = inhibitory_to_excitatory_connections_null
    for a in range(0, round((deleteRate_ItoE / 100) * (
            NUMBER_OF_CONFIGURATIONS * 50 - len(inhibitory_to_excitatory_connections_null)))):
        random_connection = PickCon(NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS, deletedCon)
        inhibitory_to_excitatory_connections[random_connection[0]][random_connection[1]] = 0
        deletedCon.append(random_connection)

    # initialize the excitatory neurons
    NumberOfConvergences = 0


    NUMBER_OF_LOOPS = NUMBER_OF_CONFIGURATIONS

    for i in range(0, NUMBER_OF_LOOPS):

        random_index = i

        for index in range(0, NUMBER_OF_NEURONS):
            if Configurations[random_index][index] == 1:
                excitatory_neurons[index] = 1
            else:
                excitatory_neurons[index] = 0

        # change the values of 10 percent of neurons

        for index in range(0, 10):
            key = PickNeron(NUMBER_OF_NEURONS)
            if excitatory_neurons[key] == 1:
                excitatory_neurons[key] = 0

            elif excitatory_neurons[key] == 0:
                excitatory_neurons[key] = 1

        # we initialize our inhibitory neurons here
        inhibitory_neurons[inhibitory_neurons != 0] = 0
        sum_list = []

        for a in range(0, NUMBER_OF_CONFIGURATIONS):
            sum = 0
            for j in range(0, NUMBER_OF_NEURONS):
                if excitatory_to_inhibitory_connections[j][a] != 0 and excitatory_neurons[j] == 1:
                    sum += 1

            sum_list.append(sum)

        inhibitory_neurons[sum_list.index(max(sum_list))] = 1

        # run the dynamics
        for m in range(0, 1000):
            tag = PickNeron(NUMBER_OF_NEURONS)
            sums = 0

            for u in range(0, NUMBER_OF_NEURONS):
                sums += excitatory_to_excitatory_connections[u][tag] * excitatory_neurons[u]

            sums += sums / NUMBER_OF_NEURONS

            for u in range(0, NUMBER_OF_CONFIGURATIONS):
                sums += inhibitory_to_excitatory_connections[u][tag] * inhibitory_neurons[u]

            if sums > 0:
                excitatory_neurons[tag] = 1

            else:
                excitatory_neurons[tag] = 0

            # update the inhibitory neurons every step
            inhibitory_neurons[inhibitory_neurons != 0] = 0
            sum_list = []
            for q in range(0, NUMBER_OF_CONFIGURATIONS):
                sum = 0

                for j in range(0, NUMBER_OF_NEURONS):
                    if excitatory_to_inhibitory_connections[j][q] != 0 and excitatory_neurons[j] == 1:
                        sum += 1

                sum_list.append(sum)

            inhibitory_neurons[sum_list.index(max(sum_list))] = 1

            ###########################################
            # check overlap in every step in dynamics
            if NUMBER_OF_CONFIGURATIONS == MAX_NUMBER_CONFIGURATIONS:
                overlaps = 0
                for k in range(0, NUMBER_OF_NEURONS):
                    # not sure about this part
                    if excitatory_neurons[k] == 1 and Configurations[random_index][k] == 1:
                        overlaps += 1
                    elif excitatory_neurons[k] == 1 and Configurations[random_index][k] == 0:
                        overlaps += (-1)
                    elif excitatory_neurons[k] == 0 and Configurations[random_index][k] == 1:
                        overlaps += (-1)
                    elif excitatory_neurons[k] == 0 and Configurations[random_index][k] == 0:
                        overlaps += 1

                # overlaps = overlaps / 500
                cmp_list[i][m] = overlaps / NUMBER_OF_NEURONS
            ########################################

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


    print(NUMBER_OF_CONFIGURATIONS, ": ", NumberOfConvergences / (NUMBER_OF_LOOPS))
    return round(NumberOfConvergences / (NUMBER_OF_LOOPS), 4)

######################################################################
if __name__ == '__main__':
    for NUMBER_OF_TEMPERATURE1 in range(0, MAX_NUMBER_TEMPERATURES1):
        Legends1 = []
        result=[]
        t = 0
        for NUMBER_OF_TEMPERATURE in range(0, MAX_NUMBER_TEMPERATURES):
            for avg_index in range(0, 3):
                for deleteRate_eTOe in [0, 25, 50, 75, 90]:
                    for deleteRate_iTOe in [0, 10, 15, 25, 50, 75]:
                        folder_name = "results"
                        if not os.path.exists(folder_name):
                            os.makedirs(folder_name)
                        # for NUMBER_OF_CONFIGURATIONS in range(100, MAX_NUMBER_CONFIGURATIONS + 1):
                        # multi process part
                        x = np.arange(start, MAX_NUMBER_CONFIGURATIONS + 1, 1)
                        y1 = [deleteRate_eTOe] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                        y2 = [deleteRate_iTOe] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                        x_y = zip(x, y1, y2)

                        pool = mp.Pool()
                        res = pool.map(multicores, list(x_y))
                        result.append(res[-1])


                        # sixth step: plot the graph
                        X1[NUMBER_OF_TEMPERATURE1] = np.arange(start, (MAX_NUMBER_CONFIGURATIONS + 1),
                                                               1) / NUMBER_OF_NEURONS
                        # draw the capacity
                        for NUMBER_OF_TEMPERATURE1 in range(0, MAX_NUMBER_TEMPERATURES1):
                            plt.plot(X1[NUMBER_OF_TEMPERATURE1], res, marker='.',
                                     label="Temperature 1=" + str(Temperatures1[NUMBER_OF_TEMPERATURE1]))
                            Legends1.append("Temperature 1=" + str(Temperatures1[NUMBER_OF_TEMPERATURE1]))

                        plt.legend()
                        file_name = folder_name + "capacities/deleteRate_eTOe=" + str(
                            deleteRate_eTOe) + "/deleteRate_iTOe=" + str(deleteRate_iTOe)

                        if not os.path.exists(file_name):
                            os.makedirs(file_name)

                        plt.xlabel("Saturation")
                        plt.ylabel("Capacity")
                        plt.savefig(file_name + "/capacity.png")
                        # plt.show()
                        plt.close()

    # show and save the deficiency matrix
    # compress matrix
    print(np.shape(result))

    # save raw result
    save_raw_data(result)

    show_matrix = np.mean(np.array(result).reshape(3, 5, 6), axis=0)
    show_matrix_std = np.std(np.array(result).reshape(3, 5, 6), axis=0)
    print(show_matrix)


    show_matrix=np.round(show_matrix, 2)
    show_matrix_std = np.round(show_matrix_std, 2)
    fig, ax = plt.subplots()
    im = ax.matshow(show_matrix, cmap=plt.cm.Blues)
    for i in range(0, 6):
        for j in range(0, 5):
            c = show_matrix[j, i]
            std_value = show_matrix_std[j, i]
            plt.text(i, j, f"{c} \n ± {std_value}", va='center', ha='center')

    file_name2 = folder_name + "/deciencies_matrix"
    plt.xlabel("Fraction of removed E-I connections (%)")
    plt.ylabel('Fraction of removed E-E connections (%)')
    plt.xticks(np.arange(0, 6, 1), [0, 10, 15, 25, 50, 75])
    plt.yticks(np.arange(0, 5, 1), [0, 25, 50, 75, 90])
    if not os.path.exists(file_name2):
        os.makedirs(file_name2)

    cbar = plt.colorbar(im)
    cbar.set_label('Recall fraction')
    plt.savefig(file_name2 + "/deficiency_matrix.png")
    plt.show()
    plt.close()