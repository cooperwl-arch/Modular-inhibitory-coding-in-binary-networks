# 2024/5/29
# original one neuron model
# but implement deficiency in the model
# no temperature
# This is the one running again for 1

import os
import json

import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import random
import math
import pandas as pd


start = 1
MAX_NUMBER_CONFIGURATIONS = 1500
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
            "raw_data_def_e_i_0.csv",
            "raw_data_def_e_i_25.csv",
            "raw_data_def_e_i_50.csv",
            "raw_data_def_e_i_75.csv"
        ],
        "parameter_varied_between_files": {
            "fraction of removed E-I connections (%)": [0, 25, 50, 75]
        },
        "value": "recall fraction"
    }

    with open(
            os.path.join(raw_data_folder, "metadata.json"),
            "w"
    ) as f:
        json.dump(metadata, f, indent=4)


def save_capacity_raw_data(res, x, def_e_i, folder_name):
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
            f"raw_data_def_e_i_{def_e_i}.csv"
        )
    )


def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def PickCon(number1, number2, deletedCon):
    list2 = [[x, y] for y in range(number2) for x in range(number1)]
    list1_set = set(map(tuple, deletedCon))
    list2 = [list(sublist) for sublist in set(map(tuple, list2)) - list1_set]
    return random.choice(list2)


def multicores(args):
    NUMBER_OF_CONFIGURATIONS, deleteRate_EtoI, deleteRate_EtoI2, deleteRate_EtoE, deleteRate_ItoE, Temperature = args
    e_i_connections_pool1 = []
    e_i_connections_pool2 = []
    e_i_connections_pool3 = []
    e_i_connections_pool4 = []
    e_i_connections_pool5 = []

    Configurations = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    # first step: initialize the configurations
    excitatory_to_excitatory_connections = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_NEURONS), dtype=float)

    inhibitory_to_excitatory_connections = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    excitatory_to_inhibitory_connections = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_CONFIGURATIONS), dtype=float)

    inhibitory_to_excitatory_connections2 = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    excitatory_to_inhibitory_connections2 = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_CONFIGURATIONS), dtype=float)

    inhibitory_to_excitatory_connections3 = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    excitatory_to_inhibitory_connections3 = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_CONFIGURATIONS), dtype=float)

    inhibitory_to_excitatory_connections4 = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    excitatory_to_inhibitory_connections4 = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_CONFIGURATIONS), dtype=float)

    inhibitory_to_excitatory_connections5 = np.zeros((NUMBER_OF_CONFIGURATIONS, NUMBER_OF_NEURONS), dtype=float)
    excitatory_to_inhibitory_connections5 = np.zeros((NUMBER_OF_NEURONS, NUMBER_OF_CONFIGURATIONS), dtype=float)

    # second step: create two layers of neurons and implement learning
    # remember there are two different layers
    excitatory_neurons = np.zeros(NUMBER_OF_NEURONS, dtype=float)
    # The number of inhibitory neurons is the number of configuration
    inhibitory_neurons = np.zeros(NUMBER_OF_CONFIGURATIONS, dtype=float)
    inhibitory_neurons2 = np.zeros(NUMBER_OF_CONFIGURATIONS, dtype=float)
    inhibitory_neurons3 = np.zeros(NUMBER_OF_CONFIGURATIONS, dtype=float)
    inhibitory_neurons4 = np.zeros(NUMBER_OF_CONFIGURATIONS, dtype=float)
    inhibitory_neurons5 = np.zeros(NUMBER_OF_CONFIGURATIONS, dtype=float)

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

            # not quite sure this part is correct
            weight = weight / (NUMBER_OF_NEURONS * NUMBER_OF_CONFIGURATIONS)

            # not quite sure this part is correct
            excitatory_to_excitatory_connections[i][j] = weight

        for k in range(0, NUMBER_OF_CONFIGURATIONS):

            if Configurations[k][i] == 1:
                # not quite sure this part is correct
                excitatory_to_inhibitory_connections[i][k] = 1 / (NUMBER_OF_NEURONS)
                excitatory_to_inhibitory_connections2[i][k] = 1 / (NUMBER_OF_NEURONS)
                excitatory_to_inhibitory_connections3[i][k] = 1 / (NUMBER_OF_NEURONS)
                excitatory_to_inhibitory_connections4[i][k] = 1 / (NUMBER_OF_NEURONS)
                excitatory_to_inhibitory_connections5[i][k] = 1 / (NUMBER_OF_NEURONS)

                e_i_connections_pool1.append([1, i, k])
                e_i_connections_pool2.append([2, i, k])
                e_i_connections_pool3.append([3, i, k])
                e_i_connections_pool4.append([4, i, k])
                e_i_connections_pool5.append([5, i, k])

            else:
                inhibitory_to_excitatory_connections[k][i] = -1
                inhibitory_to_excitatory_connections2[k][i] = -1
                inhibitory_to_excitatory_connections3[k][i] = -1
                inhibitory_to_excitatory_connections4[k][i] = -1
                inhibitory_to_excitatory_connections5[k][i] = -1


    big_connections_pool = e_i_connections_pool1 + e_i_connections_pool2 + e_i_connections_pool3 + e_i_connections_pool4 + e_i_connections_pool5
    deleted_Cons = random.sample(big_connections_pool,
                                 round((deleteRate_EtoI / 100) * len(big_connections_pool)))
    random.shuffle(deleted_Cons)

    ######################### delete connections #################################
    for a in range(0, round((deleteRate_EtoI / 100) * len(big_connections_pool))):
        if (deleted_Cons[a])[0] == 1:
            index1 = (deleted_Cons[a])[1]
            index2 = (deleted_Cons[a])[2]
            excitatory_to_inhibitory_connections[index1][index2] = 0
        elif (deleted_Cons[a])[0] == 2:
            index1 = (deleted_Cons[a])[1]
            index2 = (deleted_Cons[a])[2]
            excitatory_to_inhibitory_connections2[index1][index2] = 0
        elif (deleted_Cons[a])[0] == 3:
            index1 = (deleted_Cons[a])[1]
            index2 = (deleted_Cons[a])[2]
            excitatory_to_inhibitory_connections3[index1][index2] = 0
        elif (deleted_Cons[a])[0] == 4:
            index1 = (deleted_Cons[a])[1]
            index2 = (deleted_Cons[a])[2]
            excitatory_to_inhibitory_connections4[index1][index2] = 0
        elif (deleted_Cons[a])[0] == 5:
            index1 = (deleted_Cons[a])[1]
            index2 = (deleted_Cons[a])[2]
            excitatory_to_inhibitory_connections5[index1][index2] = 0
    ############################################################


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
        inhibitory_neurons[inhibitory_neurons != 1] = 1
        inhibitory_neurons2[inhibitory_neurons2 != 1] = 1
        inhibitory_neurons3[inhibitory_neurons3 != 1] = 1
        inhibitory_neurons4[inhibitory_neurons4 != 1] = 1
        inhibitory_neurons5[inhibitory_neurons5 != 1] = 1

        sum_list = []
        sum_list2 = []
        sum_list3 = []
        sum_list4 = []
        sum_list5 = []

        for a in range(0, NUMBER_OF_CONFIGURATIONS):
            sum1 = 0
            sum2 = 0
            sum3 = 0
            sum4 = 0
            sum5 = 0

            # not quite sure about this part. Same weight??
            # print(excitatory_to_inhibitory_connections)
            for j in range(0, NUMBER_OF_NEURONS):
                if excitatory_to_inhibitory_connections[j][a] != 0 and excitatory_neurons[j] == 1:
                    sum1 += 1

                if excitatory_to_inhibitory_connections2[j][a] != 0 and excitatory_neurons[j] == 1:
                    sum2 += 1

                if excitatory_to_inhibitory_connections3[j][a] != 0 and excitatory_neurons[j] == 1:
                    sum3 += 1

                if excitatory_to_inhibitory_connections4[j][a] != 0 and excitatory_neurons[j] == 1:
                    sum4 += 1

                if excitatory_to_inhibitory_connections5[j][a] != 0 and excitatory_neurons[j] == 1:
                    sum5 += 1

            sum_list.append(sum1)
            sum_list2.append(sum2)
            sum_list3.append(sum3)
            sum_list4.append(sum4)
            sum_list5.append(sum5)

        all_sum_list = sum_list + sum_list2 + sum_list3 + sum_list4 + sum_list5

        ################################################################
        # print(inhibitory_neurons)
        for index_group in range(NUMBER_OF_CONFIGURATIONS):
            # index_group in the index of the configuration
            for index_neuron in range(5):
                # shut down one of the inhibitory neuron or neither
                for r2 in range(len(all_sum_list)):
                    # don't compare with itself and other neurons in the same configuration group

                    if r2 == index_group or r2 == index_group + NUMBER_OF_CONFIGURATIONS or r2 == index_group + NUMBER_OF_CONFIGURATIONS*2\
                            or r2 == index_group + NUMBER_OF_CONFIGURATIONS*3 or r2 == index_group + NUMBER_OF_CONFIGURATIONS*4:
                        continue

                    r1 = index_group + NUMBER_OF_CONFIGURATIONS * index_neuron


                    max_two_which_group = []
                    max_two_which_group.append(r1 // len(sum_list))
                    max_two_which_group.append(r2 // len(sum_list))
                    if all_sum_list[r1] - all_sum_list[r2] > 0:
                        # only activate the first neuron
                        # shut down the second neuron
                        if max_two_which_group[1] == 0:
                            inhibitory_neurons[r2 % len(sum_list)] = 0

                        elif max_two_which_group[1] == 1:
                            inhibitory_neurons2[r2 % len(sum_list)] = 0

                        elif max_two_which_group[1] == 2:
                            inhibitory_neurons3[r2 % len(sum_list)] = 0

                        elif max_two_which_group[1] == 3:
                            inhibitory_neurons4[r2 % len(sum_list)] = 0

                        elif max_two_which_group[1] == 4:
                            inhibitory_neurons5[r2 % len(sum_list)] = 0

                    elif all_sum_list[r2] - all_sum_list[r1] > 0:
                        # only activate the second neuron
                        # shut down the first neuron
                        if max_two_which_group[0] == 0:
                            inhibitory_neurons[r1 % len(sum_list)] = 0

                        elif max_two_which_group[0] == 1:
                            inhibitory_neurons2[r1 % len(sum_list)] = 0

                        elif max_two_which_group[0] == 2:
                            inhibitory_neurons3[r1 % len(sum_list)] = 0

                        elif max_two_which_group[0] == 3:
                            inhibitory_neurons4[r1 % len(sum_list)] = 0

                        elif max_two_which_group[0] == 4:
                            inhibitory_neurons5[r1 % len(sum_list)] = 0


        # run the dynamics
        for m in range(0, 1000):
            tag = PickNeron(NUMBER_OF_NEURONS)
            sums = 0

            for u in range(0, NUMBER_OF_NEURONS):
                sums += excitatory_to_excitatory_connections[u][tag] * excitatory_neurons[u]

            sums += sums / NUMBER_OF_NEURONS

            for u in range(0, NUMBER_OF_CONFIGURATIONS):
                sums += inhibitory_to_excitatory_connections[u][tag] * inhibitory_neurons[u]
                sums += inhibitory_to_excitatory_connections2[u][tag] * inhibitory_neurons2[u]
                sums += inhibitory_to_excitatory_connections3[u][tag] * inhibitory_neurons3[u]
                sums += inhibitory_to_excitatory_connections4[u][tag] * inhibitory_neurons4[u]
                sums += inhibitory_to_excitatory_connections5[u][tag] * inhibitory_neurons5[u]

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
folder_name = "50 E-I deficiency/inhibi_5"

if __name__ == '__main__':
    for NUMBER_OF_TEMPERATURE1 in range(0, MAX_NUMBER_TEMPERATURES1):
        Legends1 = []
        result = []
        lower_value = []
        t = 0

        result_df = pd.DataFrame()

        for NUMBER_OF_TEMPERATURE in range(0, MAX_NUMBER_TEMPERATURES):

            #
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            tag = 0
            for def_e_i in [0, 25, 50, 75]:
                res = []
                # for NUMBER_F_CONFIGURATIONS in range(100, MAX_NUMBER_CONFIGURATIONS + 1):
                # multi process part
                for index_average in range(3):
                    x = np.arange(start, MAX_NUMBER_CONFIGURATIONS + 1, 50)
                    y0 = [def_e_i] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    y05 = [0] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    y1 = [0] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    y2 = [0] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    y3 = [Temperatures1[NUMBER_OF_TEMPERATURE1]] * (MAX_NUMBER_CONFIGURATIONS + 1 - start)
                    x_y = zip(x, y0, y05, y1, y2, y3)

                    pool = mp.Pool()
                    res.append(pool.map(multicores, list(x_y)))

                X1[NUMBER_OF_TEMPERATURE1] = np.arange(start, (MAX_NUMBER_CONFIGURATIONS + 1),
                                                       50) / NUMBER_OF_NEURONS

                average_res = np.mean([res[0], res[1], res[2]], axis=0)
                std_res = np.std([res[0], res[1], res[2]], axis=0)

                x = np.arange(start, MAX_NUMBER_CONFIGURATIONS + 1, 50)
                save_capacity_raw_data(res, x, def_e_i, folder_name)

                #                 capacities.append(res[0])
                #                 draw the capacity
                #                 for NUMBER_OF_TEMPERATURE1 in range(0, MAX_NUMBER_TEMPERATURES1):

                plt.errorbar(X1[NUMBER_OF_TEMPERATURE1], average_res, yerr=std_res, fmt='-o',
                             label="Reduction ratio=" + str(def_e_i),
                             color=(normalized_colors[tag][0], normalized_colors[tag][1], normalized_colors[tag][2]))
                Legends1.append(
                    "Reduction ratio=" + str(def_e_i))


                tag = tag + 1


    plt.legend()
    file_name = folder_name + "/capacities"

    if not os.path.exists(file_name):
        os.makedirs(file_name)

    plt.xlabel(r'Number of configurations / $n_e$')
    plt.ylabel("Recall Fraction")
    plt.ylim(-0.2, 1.2)

    # save the json file
    save_capacity_metadata(folder_name)

    # plt.title("Capacity vs Saturation for 50 ex-neurons and 5 in-neuron system")
    plt.savefig(file_name + "/capacity_50_ex-neurons_system_5_in_neuron_E-I_deficiency" + ".pdf",
                format='pdf')
    plt.savefig(file_name + "/capacity_50_ex-neurons_system_5_in_neuron_E-I_deficiency" + ".svg",
                format='svg')

    plt.show()