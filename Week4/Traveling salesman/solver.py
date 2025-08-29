#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import random
import numpy as np

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

Edge = namedtuple("Edge", ['x', 'y','u','v'])

def reduced_d_matrix_greedySolution(points, K = 20):
    node_count = len(points)
    greedy_solution = [0]
    point = 0
    M_distances = np.zeros((node_count, K), dtype=float)
    M_indexes = np.zeros((node_count, K), dtype=int)
    points_array = np.asarray(points)
    while len(greedy_solution) < len(points):
        d_vector = points_array - points_array[point]
        d_vector = np.sqrt(np.sum(d_vector*d_vector, axis=1))
        d_vector[point] = max(d_vector)
        M_indexes[point] = np.argpartition(d_vector, K)[:K].astype(int)
        M_distances[point] = d_vector[M_indexes[point]]
        d_vector[greedy_solution] = max(d_vector)
        next_point = np.argmin(d_vector)
        greedy_solution.append(next_point)
        point = next_point
    return M_distances, M_indexes, greedy_solution

# def reduced_d_matrix(points, K = 3):
#     node_count = len(points)
#     M_distances = np.zeros((node_count, K), dtype=float)
#     M_indexes = np.zeros((node_count, K), dtype=int)
#     points_array = np.asarray(points)
#     for i in range(node_count):
#         d_vector = points_array - points_array[i]
#         d_vector = np.sqrt(np.sum(d_vector*d_vector, axis=1))
#         d_vector[i] = max(d_vector)
#         M_indexes[i] = np.argpartition(d_vector, K)[:K].astype(int)
#         M_distances[i] = d_vector[M_indexes[i]]
#     return M_distances, M_indexes


def calculate_objective(points, solution):
    nodeCount = len(solution)
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])
    return obj

def two_opt(solution, idx1, d_matrix_red, d_matrix_ind, Temperature):
    # idx1 = random.randint(0, len(solution))
    next_solution = list(solution)
    # idx1 = 3

    idx2 = (idx1+1 if idx1 < len(solution)-1 else 0)
    c1, c2 = solution[idx1], solution[idx2]
    # c3 = random.choice(np.where(d_matrix[c2, :] < d_matrix[c1,c2])[0].tolist())
    # shorter_distances = np.where(d_matrix[c2, :] < d_matrix[c1,c2])[0].tolist()
    c3 = (random.choice(d_matrix_ind[c2]) if random.random() < Temperature else d_matrix_ind[c2, np.argmin(d_matrix_red[c2])])
    idx3 = solution.index(c3)
    if idx3 > idx2:
        if next_solution[idx3-1]!=c2:
            next_solution[idx3-1], next_solution[idx2] = next_solution[idx2], next_solution[idx3-1]
            next_solution[idx2+1:idx3-1] = next_solution[idx3-2:idx2:-1]
    elif c3 != c1:
        next_solution[idx3], next_solution[idx1] = next_solution[idx1], next_solution[idx3]
        next_solution[idx3+1:idx1] = next_solution[idx1-1:idx3:-1]
    return next_solution

def k_opt(points, solution, d_matrix_red, d_matrix_ind, Temperature):
    best_solution = solution
    best_result = calculate_objective(points, solution)

    # for idx1 in range(len(solution)):
    idx1 = random.randint(0, len(solution)-1)
    different = True
    iter = 0
    while different and iter<10:
        iter+=1
        next_solution = two_opt(solution, idx1, d_matrix_red, d_matrix_ind, Temperature)
        next_result = calculate_objective(points, next_solution)
        if next_solution == solution:
            different = False
        else:
            if next_result < best_result:
                best_solution = next_solution
                best_result = next_result
            solution = next_solution
            # draw_trajetory(points, solution)
    return best_solution, best_result

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    d_matrix_red, d_matrix_ind, greedy_solution = reduced_d_matrix_greedySolution(points)

    best_solution = solution = greedy_solution
    best_result = calculate_objective(points, greedy_solution)

    number_iterations = 100000
    for iteration in range(number_iterations):
        Temperature = max(0.7, np.exp(-(1 / 500) * iteration))
        solution, result = k_opt(points, solution, d_matrix_red, d_matrix_ind, Temperature)
        if result < best_result:
            best_result = result
            best_solution = solution

        print("iteration {}".format(iteration))
        print('best resutlt {}'.format(result))

    obj = length(points[best_solution[-1]], points[best_solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[best_solution[index]], points[best_solution[index + 1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_solution))

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

