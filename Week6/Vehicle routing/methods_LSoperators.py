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

def route_concat(vehicle_tours, df_customers, distance_matrix, vehicle_capacity):
    number_routes = len(vehicle_tours)
    gains_dict = {}
    max_gain = 0
    best_concatenate = (0,0)
    for i in range(number_routes):
        for j in range(number_routes):
            if i == j:
                gains_dict[(i,j)]=-np.inf
            else:
                gain = distance_matrix[int(vehicle_tours[i][-1]),0] + distance_matrix[0, int(vehicle_tours[j][0])] \
                - distance_matrix[int(vehicle_tours[i][-1]), int(vehicle_tours[j][0])]
                gains_dict[(i, j)] = gain
                if gain>max_gain and sum(df_customers['demand'].loc[df_customers['serv_vehicle'].isin([i]+[j])])<=vehicle_capacity:
                    best_concatenate = (i,j)
                    max_gain=gain
    if max_gain > 0:
        df_customers['next_customer'].loc[int(vehicle_tours[best_concatenate[0]][-1])] = int(
            vehicle_tours[best_concatenate[1]][0])
        df_customers['previous_customer'].loc[int(vehicle_tours[best_concatenate[1]][0])] = int(
            vehicle_tours[best_concatenate[0]][-1])
        df_customers['serv_vehicle'].loc[df_customers['serv_vehicle'].isin([best_concatenate[1]])] = best_concatenate[0]
        df_customers['serv_vehicle'].loc[df_customers['serv_vehicle']>best_concatenate[1]] -= 1
        vehicle_tours[best_concatenate[0]] = vehicle_tours[best_concatenate[0]] + vehicle_tours[best_concatenate[1]]
        vehicle_tours.remove(vehicle_tours[best_concatenate[1]])


    return vehicle_tours, df_customers





