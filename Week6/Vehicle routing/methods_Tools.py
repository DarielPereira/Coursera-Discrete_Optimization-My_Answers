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

### Method compute distance Matrix
def d_matrix(locations):
    node_count = len(locations)
    M_distances = np.zeros((node_count, node_count), dtype=float)
    points_array = np.asarray(locations)
    for i in range(node_count):
        d_vector = points_array - points_array[i]
        d_vector = np.sqrt(np.sum(d_vector*d_vector, axis=1))
        M_distances[i] = d_vector
    return M_distances

### Method draw map
def draw_problem(depot, df_customers):
    plt.scatter(list(df_customers.x), list(df_customers.y), c = list(df_customers['serv_vehicle']), marker='*')
    # plt.xlim(min(df_customers.locationx),max(df_customers.locationx))
    # plt.ylim(min(df_customers.locationy), max(df_customers.locationy))
    plt.scatter(depot.x, depot.y, c = -1, marker='d')
    plt.show()

### Method Compute Objective
def computeObjective(vehicle_tours, M):
    obj = 0
    for v in range(0, len(vehicle_tours)):
        vehicle_tour = ['0']+vehicle_tours[v]+['0']
        if len(vehicle_tour) > 2:
            for i in range(0, len(vehicle_tour)-1):
                obj += M[int(vehicle_tour[i]),int(vehicle_tour[i+1])]
    return obj

def add_zerosTour(vehicle_tours):
    for i in range(len(vehicle_tours)):
        vehicle_tours[i] = ['0'] + vehicle_tours[i] + ['0']
    return vehicle_tours


def remove_zerosTour(vehicle_tours):
    for i in range(len(vehicle_tours)):
        del vehicle_tours[i][0]
        del vehicle_tours[i][-1]
    return vehicle_tours


