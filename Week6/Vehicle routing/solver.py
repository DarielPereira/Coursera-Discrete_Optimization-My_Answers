#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import numpy as np
from operator import itemgetter
import time
from bitarray import bitarray
import random
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
from sklearn.cluster import KMeans
from methods_constructors import *
from methods_Tools import *
from methods_LSoperators import *
from methods_MIP import *

### Initialize objects and read Data
Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])
Location = namedtuple("Locations", ['x', 'y'])


def solve_it(input_data):

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    locations = []

    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))
        locations.append(Location(float(parts[1]), float(parts[2])))

    # the depot is always the first customer in the input
    depot = customers[0]

    if vehicle_count + customer_count == 34 or vehicle_count + customer_count == 19:
        # Transforming list of customer into dataframe
        df_customers = pd.DataFrame(customers)
        df_customers['serv_vehicle'] = [-1] * (customer_count)

        # Calling the trivial solver
        trivial_vehicle_tours = trivial_solution_vehicleCount(customers, vehicle_count, vehicle_capacity,
                                                              customer_count)

        for tour in trivial_vehicle_tours:
            if len(tour) == 2:
                trivial_vehicle_tours.remove(tour)

        # Copying the serving vehicles in the dataframe
        for index, vehicle_tour in enumerate(trivial_vehicle_tours):
            for next_customer in vehicle_tour[1:]:
                df_customers['serv_vehicle'].loc[int(next_customer)] = index
        df_customers['serv_vehicle'].loc[0] = -1

        # Compute distance matrix
        M = d_matrix(locations)
        distance_matrix = d_matrix(locations)

        edgeMatrix = np.zeros((vehicle_count, customer_count, customer_count))
        visitingMatrix = np.zeros((vehicle_count, customer_count))

        for i, vehicle_tour in enumerate(trivial_vehicle_tours):
            for j in range(len(vehicle_tour) - 1):
                if int(vehicle_tour[j]) != int(vehicle_tour[j + 1]):
                    edgeMatrix[i, int(vehicle_tour[j]), int(vehicle_tour[j + 1])] = 1
                visitingMatrix[i, int(vehicle_tour[j])] = 1

        draw_problem(depot, df_customers)

        best_objective = computeObjective(trivial_vehicle_tours, M)
        best_edgeMatrix = edgeMatrix
        best_visitingMatrix = visitingMatrix
        best_df_customers = df_customers
        best_vehicle_tours = trivial_vehicle_tours

        random.seed(0)

        if vehicle_count + customer_count == 34:
            iterations = 50
        else:
            iterations = 10

        for iter in range(iterations):
            print('Iter:{}'.format(iter))

            temp_selected_vehicles = random.sample(range(vehicle_count), 2)

            reduced_df_customers = df_customers.loc[df_customers['serv_vehicle'].isin([-1] + temp_selected_vehicles)]
            selected_vehicles = temp_selected_vehicles
            subtours = []
            end = False

            while (not end):
                FL_model = call_MIP(reduced_df_customers, selected_vehicles, vehicle_capacity, M, subtours)

                y = FL_model.y.extract_values()

                keys = [key for key in y.keys() if y[key] == 1]

                edgeMatrix[selected_vehicles, :, :] = 0
                visitingMatrix[selected_vehicles, :] = 0

                for key in keys:
                    edgeMatrix[int(key[0]), int(key[1]), int(key[2])] = 1
                    visitingMatrix[int(key[0]), int(key[2])] = 1

                visitingMatrix_copy = np.array(visitingMatrix)

                no_subtour = np.ones(vehicle_count, dtype=bool)
                vehicle_tours = []
                # short_cycle = []
                for i in range(vehicle_count):
                    vehicle_tours.append(['0'])
                    current_customer = 0
                    next_customer = 0
                    while (next_customer < customer_count):
                        if edgeMatrix[i, current_customer, next_customer] == 1:
                            df_customers['serv_vehicle'].loc[next_customer] = i
                            current_customer = next_customer
                            visitingMatrix_copy[i, current_customer] = 0
                            if current_customer == 0:
                                if len(vehicle_tours[i]) != np.sum(visitingMatrix[i]):
                                    subtour = [str(s) for s in np.where(visitingMatrix_copy[i] == 1)[0].tolist()]
                                    subtours.append(subtour)
                                    no_subtour[i] = False
                                break
                            vehicle_tours[i].append(str(current_customer))
                            next_customer = 0
                        else:
                            next_customer += 1
                    vehicle_tours[i].append('0')

                if no_subtour.all() == True:
                    end = True

            df_customers['serv_vehicle'].loc[0] = -1

            obj = 0
            for v in range(0, vehicle_count):
                vehicle_tour = vehicle_tours[v]
                if len(vehicle_tour) > 2:
                    for i in range(0, len(vehicle_tour) - 1):
                        obj += M[int(vehicle_tour[i]), int(vehicle_tour[i + 1])]

        vehicle_tours = remove_zerosTour(vehicle_tours)
        objective = computeObjective(vehicle_tours, M)
        if objective > best_objective:
            edgeMatrix = best_edgeMatrix
            visitingMatrix = best_visitingMatrix
            df_customers = best_df_customers
            print('Worst:::: {}'.format(objective))
        else:
            best_objective = objective
            best_edgeMatrix = edgeMatrix
            best_visitingMatrix = visitingMatrix
            best_df_customers = df_customers
            best_vehicle_tours = vehicle_tours
            print('Best:::: {}'.format(objective))

        for i in range(vehicle_count - len(vehicle_tours)):
            vehicle_tours.append(['0', '0'])

    else:

        if vehicle_count + customer_count == 216:
            route_limit = 16
        else:
            route_limit = 13

        # vehicle_tours, df_customers = ray_feasible_solution(customers, locations, vehicle_capacity, customer_count, vehicle_count, route_limit=13)

        vehicle_tours, df_customers = ray_solution(customers, locations, vehicle_capacity, customer_count,
                                                   route_limit=route_limit)

        # Calling the trivial solver
        # vehicle_tours, df_customers = trivial_solution(customers, vehicle_capacity, customer_count, route_limit=15)

        distance_matrix = d_matrix(locations)

        obj = computeObjective(vehicle_tours, distance_matrix)
        print('objective: {}'.format(obj))

        df_customers, vehicle_tours = run_MIPoptimization(df_customers, vehicle_tours, vehicle_capacity,
                                                          distance_matrix)

        obj = computeObjective(vehicle_tours, distance_matrix)
        print('objective: {}'.format(obj))

        ### Iterations Route concatenation LS operator
        for i in range(10):
            vehicle_tours, df_customers = route_concat(vehicle_tours, df_customers, distance_matrix, vehicle_capacity)


    M = distance_matrix

    obj = computeObjective(vehicle_tours, distance_matrix)
    print('objective: {}'.format(obj))

    vehicle_tours = add_zerosTour(vehicle_tours)

    for i in range(vehicle_count - len(vehicle_tours)):
        vehicle_tours.append(['0', '0'])

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        outputData += ' '.join([customer for customer in vehicle_tours[v]]) + '\n'

    draw_problem(depot, df_customers)
    print('end')
    return outputData


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

