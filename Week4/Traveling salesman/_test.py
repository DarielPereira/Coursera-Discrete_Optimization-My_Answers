import math
from collections import namedtuple
import numpy as np
from operator import itemgetter
import time
from bitarray import bitarray
import random
import matplotlib.pyplot as plt

# input_location = '././data/tsp_574_1'
input_location = '././data/tsp_Kopt'

with open(input_location, 'r') as input_data_file:
    input_data = input_data_file.read()
# print(input_data)

Point = namedtuple("Point", ['x', 'y'])
Edge = namedtuple("Edge", ['x', 'y','u','v'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def dist_matrix(points):
    node_count = len(points)
    M = np.zeros((node_count, node_count), dtype=float)
    for idx, point in enumerate(points):
        for idx2, point2 in enumerate(points):
            M[idx,idx2] = M[idx2,idx] = length(point, point2)
            if idx == idx2:
                M[idx,idx2] = 1000000
    return M.round(2)

def draw_point(points):
    plt.scatter(*zip(*points))
    plt.xlim(-0.1, 2)
    plt.ylim(-0.1, 2)

def calculate_objective(points, solution):
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])
    return obj

def draw_trajetory(points, solution):
    node_count = len(points)
    solution_graph = []
    start_point = points[solution[0]]

    for index in range(1, node_count):
        solution_graph += [Edge(start_point.x, start_point.y, points[solution[index]].x - start_point.x, points[solution[index]].y-start_point.y)]
        start_point = points[solution[index]]
    solution_graph += [Edge(start_point.x, start_point.y, points[solution[0]].x-start_point.x, points[solution[0]].y-start_point.y)]

    draw_point(points)

    plt.quiver(*zip(*solution_graph), color='r',angles='xy', scale_units='xy',  scale=1)
    plt.autoscale(enable=True, axis='both', tight=None)
    texts = range(node_count)

    for i, txt in enumerate(texts):
        plt.annotate(txt, (points[i].x, points[i].y+0.1))

    plt.show()

def two_opt(solution, idx1, d_matrix, Temperature):
    # idx1 = random.randint(0, len(solution))
    next_solution = list(solution)
    # idx1 = 3

    idx2 = (idx1+1 if idx1 < len(solution)-1 else 0)
    c1, c2 = solution[idx1], solution[idx2]
    # c3 = random.choice(np.where(d_matrix[c2, :] < d_matrix[c1,c2])[0].tolist())
    shorter_distances = np.where(d_matrix[c2, :] < d_matrix[c1,c2])[0].tolist()
    c3 = (random.choice(shorter_distances) if random.random() < Temperature and len(shorter_distances) > 0 else np.argmin(d_matrix[c2, :]))
    idx3 = solution.index(c3)
    if idx3 > idx2:
        if next_solution[idx3-1]!=c2:
            next_solution[idx3-1], next_solution[idx2] = next_solution[idx2], next_solution[idx3-1]
            next_solution[idx2+1:idx3-1] = next_solution[idx3-2:idx2:-1]
    elif c3 != c1:
        next_solution[idx3], next_solution[idx1] = next_solution[idx1], next_solution[idx3]
        next_solution[idx3+1:idx1] = next_solution[idx1-1:idx3:-1]
    return next_solution

def k_opt(points, solution, d_matrix, Temperature):
    best_solution = solution
    best_result = calculate_objective(points, solution)

    for idx1 in range(len(solution)):
        different = True
        iter = 0
        while different and iter<10:
            iter+=1
            next_solution = two_opt(solution, idx1, d_matrix, Temperature)
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

lines = input_data.split('\n')

nodeCount = int(lines[0])

points = []
for i in range(1, nodeCount+1):
    line = lines[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

d_matrix = dist_matrix(points)
print(d_matrix)


solution = [a for a in range(nodeCount)]
random.seed(0)
random.shuffle(solution)

print('initial trajectory')
draw_trajetory(points, solution)

best_solution = solution
best_result = calculate_objective(points, solution)

number_iterations = 100
for iteration in range(number_iterations):
    Temperature = max(0.2, np.exp(-(1/500)*iteration))
    solution, result = k_opt(points, best_solution, d_matrix, Temperature)
    if result < best_result:
        best_result = result
        best_solution = solution

    print("iteration {}".format(iteration))
    print('best resutlt {}'.format(result))

print('one iteration K-opt')
draw_trajetory(points, best_solution)

obj = length(points[best_solution[-1]], points[best_solution[0]])
for index in range(0, nodeCount-1):
    obj += length(points[best_solution[index]], points[best_solution[index+1]])

# prepare the solution in the specified output format
output_data = '%.2f' % obj + ' ' + str(0) + '\n'
output_data += ' '.join(map(str, best_solution))

print(output_data)
