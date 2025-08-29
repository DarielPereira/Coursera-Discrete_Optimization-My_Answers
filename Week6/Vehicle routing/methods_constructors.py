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



### Method Trivial solution
def trivial_solution_vehicleCount(customers, vehicle_count, vehicle_capacity, customer_count):
    # the depot is always the first customer in the input
    depot = customers[0]

    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []

    remaining_customers = set(customers)
    remaining_customers.remove(depot)

    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append(['0'])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand * customer_count + customer.index)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(str(customer.index))
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used
        vehicle_tours[v].append('0')



    # # checks that the number of customers served is correct
    # assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    return vehicle_tours



###Trivial Solution Course
#Route-driven: vehicles are added to current route until is full
def trivial_solution(customers, vehicle_capacity, customer_count, route_limit=np.inf):
    df_customers = pd.DataFrame(customers)
    df_customers['serv_vehicle'] = [-1] * (customer_count)
    df_customers['next_customer'] = [-1] * (customer_count)
    df_customers['previous_customer'] = [-1] * (customer_count)

    # the depot is always the first customer in the input
    depot = customers[0]

    # build a trivial solution
    vehicle_tours = []

    remaining_customers = set(customers)
    remaining_customers.remove(depot)

    v = 0
    while len(remaining_customers)!= 0:
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        bool_route_limit = False
        previous_vehicle = 0
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0 and bool_route_limit == False:
            order = sorted(remaining_customers, key=lambda customer: -customer.demand * customer_count + customer.index)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(str(customer.index))
                    remaining_customers.remove(customer)
                    df_customers['serv_vehicle'].loc[customer.index] = v
                    df_customers['next_customer'].loc[previous_vehicle] = customer.index
                    df_customers['previous_customer'].loc[customer.index] = previous_vehicle
                    previous_vehicle = customer.index
                if len(vehicle_tours[v])==route_limit:
                    df_customers['next_customer'].loc[previous_vehicle] = 0
                    bool_route_limit = True
                    break
        df_customers['next_customer'].loc[previous_vehicle] = 0
        v += 1

    return vehicle_tours, df_customers


###Ray Sweeping Solution Course
#Route-driven: vehicles are added to current route until is full
def ray_solution(customers, locations, vehicle_capacity, customer_count, route_limit=np.inf):
    df_customers = pd.DataFrame(customers)
    df_customers['serv_vehicle'] = [-1] * (customer_count)
    df_customers['next_customer'] = [-1] * (customer_count)
    df_customers['previous_customer'] = [-1] * (customer_count)
    df_customers['angle'] = [-1] * (customer_count)

    node_count = len(locations)
    points_array = np.asarray(locations)
    adjusted_points_array = points_array - points_array[0,:]
    norms = np.sqrt(np.sum(adjusted_points_array*adjusted_points_array, axis=1))
    angles = np.degrees(np.angle(adjusted_points_array[:,0]+1j*adjusted_points_array[:,1]))
    angles[angles < 0] = angles[angles < 0] + 360

    df_customers['angle'] = angles

    df_customers = df_customers.sort_values(by=['angle'], ascending=False)
    remaining_customers = df_customers.index.to_list()
    vehicle_tours = []

    remaining_customers.remove(0)

    v = 0
    while len(remaining_customers) != 0:
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        bool_route_limit = False
        previous_vehicle = 0
        while sum([capacity_remaining >= df_customers['demand'].loc[customer] for customer in
                   remaining_customers]) > 0 and bool_route_limit == False:
            i = 0
            while i < len(remaining_customers):
                customer = remaining_customers[i]
                if capacity_remaining >= df_customers['demand'].loc[customer]:
                    capacity_remaining -= df_customers['demand'].loc[customer]
                    vehicle_tours[v].append(str(customer))
                    remaining_customers.remove(customer)
                    df_customers['serv_vehicle'].loc[customer] = v
                    df_customers['next_customer'].loc[previous_vehicle] = customer
                    df_customers['previous_customer'].loc[customer] = previous_vehicle
                    previous_vehicle = customer
                    i = 0
                else:
                    i += 1
                if len(vehicle_tours[v]) == route_limit:
                    df_customers['next_customer'].loc[previous_vehicle] = 0
                    bool_route_limit = True
                    break
        df_customers['next_customer'].loc[previous_vehicle] = 0
        v += 1

    return vehicle_tours, df_customers


###Ray Sweeping Solution Course
#Route-driven: vehicles are added to current route until is full
def ray_feasible_solution(customers, locations, vehicle_capacity, customer_count, vehicle_count, route_limit=np.inf):
    df_customers = pd.DataFrame(customers)
    df_customers['serv_vehicle'] = [-1] * (customer_count)
    df_customers['next_customer'] = [-1] * (customer_count)
    df_customers['previous_customer'] = [-1] * (customer_count)
    df_customers['angle'] = [-1] * (customer_count)

    node_count = len(locations)
    points_array = np.asarray(locations)
    adjusted_points_array = points_array - points_array[0,:]
    norms = np.sqrt(np.sum(adjusted_points_array*adjusted_points_array, axis=1))
    angles = np.degrees(np.angle(adjusted_points_array[:,0]+1j*adjusted_points_array[:,1]))
    angles[angles < 0] = angles[angles < 0] + 360

    df_customers['angle'] = angles

    vehicle_tours = []
    vehicle_capacities = []

    for i in range(vehicle_count):
        vehicle_tours.append([])
        vehicle_capacities.append([])

    df_customers = df_customers.sort_values(by=['demand'], ascending=False)
    highDemanding_customers = df_customers.index.to_list()[0:vehicle_count]
    highDemanding_customers = df_customers.loc[highDemanding_customers].sort_values(by=['angle']).index.to_list()
    df_customers['previous_customer'].loc[df_customers.index.isin(highDemanding_customers)] = 0

    for i in range(vehicle_count):
        vehicle_tours[i].append(str(highDemanding_customers[i]))
        vehicle_capacities[i] = vehicle_capacity - df_customers['demand'].loc[highDemanding_customers[i]]
        df_customers['serv_vehicle'].loc[highDemanding_customers[i]] = i

    df_customers = df_customers.sort_values(by=['angle'])
    remaining_customers = df_customers.index.to_list()
    remaining_customers = [elem for elem in remaining_customers if elem not in highDemanding_customers]
    remaining_customers.remove(0)

    v = 0
    while len(remaining_customers) != 0:
        # print "Start Vehicle: ",v
        bool_route_limit = False
        previous_vehicle = int(vehicle_tours[v][-1])
        while sum([vehicle_capacities[v] >= df_customers['demand'].loc[customer] for customer in
                   remaining_customers]) > 0 and bool_route_limit == False:
            i = 0
            while i < len(remaining_customers):
                customer = remaining_customers[i]
                if vehicle_capacities[v] >= df_customers['demand'].loc[customer]:
                    vehicle_capacities[v] -= df_customers['demand'].loc[customer]
                    vehicle_tours[v].append(str(customer))
                    remaining_customers.remove(customer)
                    df_customers['serv_vehicle'].loc[customer] = v
                    df_customers['next_customer'].loc[previous_vehicle] = customer
                    df_customers['previous_customer'].loc[customer] = previous_vehicle
                    previous_vehicle = customer
                    i = 0
                else:
                    i += 1
                if len(vehicle_tours[v]) == route_limit:
                    df_customers['next_customer'].loc[previous_vehicle] = 0
                    bool_route_limit = True
                    v+=1
                    break
        df_customers['next_customer'].loc[previous_vehicle] = 0
        v += 1

    return vehicle_tours, df_customers