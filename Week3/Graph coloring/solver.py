#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import random
import time


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    init = time.time()

    node_degree = np.zeros((node_count, 1))
    for par in edges:
        node_degree[par[0]] += 1
        node_degree[par[1]] += 1

    # print(node_degree)

    edge_matrix = np.zeros((node_count, node_count), dtype=bool)

    for par in edges:
        edge_matrix[par[0], par[1]] = 1

    edge_matrix = edge_matrix + edge_matrix.T
    # print(edge_matrix)

    color_counter = 0
    color = np.zeros((node_count, 1), dtype=bool)
    temp_color = np.zeros((node_count, 1), dtype=bool)
    colors = [np.zeros((node_count, 1), dtype=bool)]

    colored_nodes = np.zeros((node_count, 1), dtype=bool)
    for elem in colors:
        colored_nodes += elem

    while (sum(colored_nodes) < node_count):
        # print(color_counter)
        max_node = np.argmax((node_degree) * np.invert(colored_nodes))
        temp_color = np.zeros((node_count, 1), dtype=bool)
        temp_color[max_node] = 1
        not_neighbors = np.invert(edge_matrix[max_node, :]).reshape(node_count, 1) * np.invert(colored_nodes)
        not_neighbors[max_node] = 0
        not_neighbors_list = (np.where(not_neighbors == 1)[0]).tolist()

        while len(not_neighbors_list) != 0:
            checking = edge_matrix[not_neighbors.flatten()] @ not_neighbors
            if sum(checking) == 0:
                temp_color = temp_color + not_neighbors
                not_neighbors_list = []
            else:
                max_node = np.argmax((node_degree) * not_neighbors)
                # not_neighbors[max_node] = 0
                temp_color[max_node] = 1
                not_neighbors = not_neighbors * np.invert(edge_matrix[:, max_node]).reshape(node_count, 1)
                not_neighbors[max_node] = 0
                not_neighbors_list = (np.where(not_neighbors == 1)[0]).tolist()
        color = temp_color
        colors.append(color)
        color_counter += 1
        colored_nodes += color
        not_neighbors_list = []

    node_colors = [0] * node_count

    colors_table = np.array(colors).reshape(color_counter + 1, node_count)

    for i in range(node_count):
        node_colors[i] = np.argmax(colors_table[:, i]) - 1

    color_upper_bound = np.max(node_colors) + 1
    print('color upper bound: {}'.format(color_upper_bound))
    # print(node_colors)

    #######################################################################
    color_upper_bound = np.max(node_colors) + 1

    best_color_optim = color_upper_bound
    best_node_colors = node_colors

    if color_upper_bound != 18:
        while best_color_optim >= color_upper_bound or time.time() - init < 600:
            color_counter = 0
            color = np.zeros((node_count, 1), dtype=bool)
            temp_color = np.zeros((node_count, 1), dtype=bool)
            colors = [np.zeros((node_count, 1), dtype=bool)]

            colored_nodes = np.zeros((node_count, 1), dtype=bool)
            for elem in colors:
                colored_nodes += elem

            while (sum(colored_nodes) < node_count):
                # print(color_counter)
                max_node = random.choice((np.where(colored_nodes == 0)[0]).tolist())
                temp_color = np.zeros((node_count, 1), dtype=bool)
                temp_color[max_node] = 1
                not_neighbors = np.invert(edge_matrix[max_node, :]).reshape(node_count, 1) * np.invert(colored_nodes)
                not_neighbors[max_node] = 0
                not_neighbors_list = (np.where(not_neighbors == 1)[0]).tolist()

                while len(not_neighbors_list) != 0:
                    checking = edge_matrix[not_neighbors.flatten()] @ not_neighbors
                    if sum(checking) == 0:
                        temp_color = temp_color + not_neighbors
                        not_neighbors_list = []
                    else:
                        max_node = random.choice((np.where(not_neighbors == 1)[0]).tolist())
                        # not_neighbors[max_node] = 0
                        temp_color[max_node] = 1
                        not_neighbors = not_neighbors * np.invert(edge_matrix[:, max_node]).reshape(node_count, 1)
                        not_neighbors[max_node] = 0
                        not_neighbors_list = (np.where(not_neighbors == 1)[0]).tolist()
                color = temp_color
                colors.append(color)
                color_counter += 1
                colored_nodes += color
                not_neighbors_list = []

            node_colors = [0] * node_count

            colors_table = np.array(colors).reshape(color_counter + 1, node_count)

            for i in range(node_count):
                node_colors[i] = np.argmax(colors_table[:, i]) - 1

            color_optim = np.max(node_colors) + 1
            print('color optim: {}'.format(color_optim))

            if color_optim <= best_color_optim:
                best_color_optim = color_optim
                best_node_colors = node_colors

    # print('best color optim: {}'.format(best_color_optim))
    # print('best node colors: {}'.format(best_node_colors))

    # prepare the solution in the specified output format
    output_data = str(best_color_optim) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_node_colors))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

