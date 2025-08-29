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
from methods_Tools import *


### Method VRP MIP Model
def create_VRP_model(vehicle_count, customer_count, distance_matrix, demand_customers, capacity_vehicles, subtour, subtour_count, subtour_lengths):
    model = pyo.ConcreteModel(name='VRP')

    model.y = pyo.Var(vehicle_count, customer_count, customer_count, within=pyo.Binary)
    model.x = pyo.Var(vehicle_count, within=pyo.Binary)

    def obj_rule(mdl):
        return sum(mdl.y[w,a,b]*distance_matrix[a,b] for w in vehicle_count for a in customer_count for b in customer_count)
    model.obj = pyo.Objective(rule=obj_rule)

    def avoid_subtour(mdl, w, s):
        if len(subtour_count)>0:
            return sum(mdl.y[w, a, b] for a in subtour[s] for b in subtour[s]) <= subtour_lengths[s] - 1
        else:
            return sum(mdl.y[w, '0', '0'] for w in vehicle_count) >= 0
    model.avoid_subtour =  pyo.Constraint(vehicle_count, subtour_count, rule = avoid_subtour)

    def vehicle_usage(mdl, w, a, b):
        return mdl.y[w, a, b]  <= mdl.x[w]
    model.vehicle_usage =  pyo.Constraint(vehicle_count, customer_count, customer_count, rule = vehicle_usage)

    def vehicle_capacity(mdl, w):
        return sum(mdl.y[w, a, b]*demand_customers[b] for a in customer_count for b in customer_count) <= capacity_vehicles
    model.vehicle_capacity =  pyo.Constraint(vehicle_count, rule = vehicle_capacity)
    #
    def unique_destination(mdl, b):
        return sum(mdl.y[w, a, b] for w in vehicle_count for a in customer_count) == 1
    model.unique_destination =  pyo.Constraint(customer_count[1:], rule = unique_destination)

    def no_stay(mdl, w, a):
        return mdl.y[w, a, a] == 0
    model.no_stay = pyo.Constraint(vehicle_count, customer_count, rule = no_stay)

    def unique_origin(mdl, a):
        return sum(mdl.y[w, a, b] for w in vehicle_count for b in customer_count) == 1
    model.unique_origin =  pyo.Constraint(customer_count[1:], rule = unique_origin)

    def start_end(mdl, w):
        return sum(mdl.y[w,'0',b] for b in customer_count)+sum(mdl.y[w,a,'0'] for a in customer_count)==2*mdl.x[w]
    model.start_end = pyo.Constraint(vehicle_count, rule = start_end)

    def in_out(mdl, w, b):
        return sum(mdl.y[w, a, b] for a in customer_count) == sum(mdl.y[w, b, a] for a in customer_count)
    model.in_out =  pyo.Constraint(vehicle_count, customer_count, rule = in_out)

    # def one_way(mdl, w, a, b):
    #     return mdl.y[w,a,b]+mdl.y[w,b,a]<=mdl.x[w]
    # model.one_way = pyo.Constraint(vehicle_count, customer_count, customer_count, rule = one_way)

    def one_way(mdl, w, a, b):
        return mdl.y[w,a,b]+mdl.y[w,b,a]<=mdl.x[w]
    model.one_way = pyo.Constraint(vehicle_count, customer_count[1:], customer_count[1:], rule = one_way)

    return model


def call_MIP(df_customers, selected_vehicles, vehicle_capacity, M, subtour):
    customer_indexes = list(df_customers.index.map(str))
    customer_indexes.sort()
    # vehicle_indexes = [str(x) for x in selected_vehicles]
    vehicle_indexes = [str(x) for x in selected_vehicles]

    subtour_indexes = [str(x) for x in list(range(len(subtour)))]

    # short_cycle_indexes = [str(x) for x in short_cycle]
    # short_cycle_large = len(short_cycle_indexes)

    distance_matrix_dict = {(a, b): M[int(a), int(b)] for a in customer_indexes for b in
                            customer_indexes}
    demand_customers = {b: df_customers['demand'].loc[int(b)] for b in customer_indexes}
    subtour_dict = {s: subtour[int(s)] for s in subtour_indexes}
    subtour_lengths = {s: len(subtour[int(s)]) for s in subtour_indexes}
    capacity_vehicles = vehicle_capacity

    FL_model = create_VRP_model(vehicle_indexes, customer_indexes, distance_matrix_dict, demand_customers,
                                capacity_vehicles, subtour_dict, subtour_indexes, subtour_lengths)

    solver = pyo.SolverFactory('glpk')

    res = solver.solve(FL_model, timelimit=300)

    return FL_model


def call_MIP_int(df_customers, selected_vehicles, vehicle_capacity, M, subtour):
    customer_indexes = list(df_customers.index.map(str))
    customer_indexes.sort()
    # vehicle_indexes = [str(x) for x in selected_vehicles]
    vehicle_indexes = [str(selected_vehicles)]

    subtour_indexes = [str(x) for x in list(range(len(subtour)))]

    # short_cycle_indexes = [str(x) for x in short_cycle]
    # short_cycle_large = len(short_cycle_indexes)

    distance_matrix_dict = {(a, b): M[int(a), int(b)] for a in customer_indexes for b in
                            customer_indexes}
    demand_customers = {b: df_customers['demand'].loc[int(b)] for b in customer_indexes}
    subtour_dict = {s: subtour[int(s)] for s in subtour_indexes}
    subtour_lengths = {s: len(subtour[int(s)]) for s in subtour_indexes}
    capacity_vehicles = vehicle_capacity

    FL_model = create_VRP_model(vehicle_indexes, customer_indexes, distance_matrix_dict, demand_customers,
                                capacity_vehicles, subtour_dict, subtour_indexes, subtour_lengths)

    solver = pyo.SolverFactory('glpk')

    res = solver.solve(FL_model, timelimit=300)

    return FL_model

def run_MIPoptimization(df_customers, vehicle_tours, vehicle_capacity, distance_matrix):
    vehicle_tours = add_zerosTour(vehicle_tours)
    customer_count = len(df_customers)
    vehicle_count = len(vehicle_tours)

    edgeMatrix = np.zeros((vehicle_count, customer_count, customer_count))
    visitingMatrix = np.zeros((vehicle_count, customer_count))

    for i, vehicle_tour in enumerate(vehicle_tours):
        for j in range(len(vehicle_tour) - 1):
            if int(vehicle_tour[j]) != int(vehicle_tour[j + 1]):
                edgeMatrix[i, int(vehicle_tour[j]), int(vehicle_tour[j + 1])] = 1
            visitingMatrix[i, int(vehicle_tour[j])] = 1

    for selected_vehicle in range(vehicle_count):
        print('selected vehicle: {}'.format(selected_vehicle))
        reduced_df_customers = df_customers.loc[df_customers['serv_vehicle'].isin([-1] + [selected_vehicle])]
        subtours = []
        end = False

        while (not end):
            FL_model = call_MIP_int(reduced_df_customers, selected_vehicle, vehicle_capacity, distance_matrix, subtours)
            y = FL_model.y.extract_values()

            keys = [key for key in y.keys() if y[key] == 1]

            edgeMatrix[selected_vehicle, :, :] = 0
            visitingMatrix[selected_vehicle, :] = 0

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

    vehicle_tours = remove_zerosTour(vehicle_tours)
    return df_customers, vehicle_tours