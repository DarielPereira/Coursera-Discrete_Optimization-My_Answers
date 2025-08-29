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
def reduced_d_matrix(locations):
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
    plt.scatter(list(df_customers.x), list(df_customers.y), c = list(df_customers['serv_Facility']), marker='*')
    # plt.xlim(min(df_customers.locationx),max(df_customers.locationx))
    # plt.ylim(min(df_customers.locationy), max(df_customers.locationy))
    plt.scatter(depot.x, depot.y, c = -1, marker='d')
    plt.show()

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

def create_MIP_reorganizeModel(vehicle_count, customer_count, demand_customers, capacity_vehicles):
    model = pyo.ConcreteModel(name='MIP_reoganize')

    model.y = pyo.Var(vehicle_count, customer_count, within=pyo.Binary)

    def capacity(mdl, w):
        return sum(mdl.y[w, b] for b in customer_count) <= vehicle_capacity
    model.capacity = pyo.Constraint(vehicle_count, rule = capacity)

    def served(mdl, b):
        return sum(mdl.y[w, b] for w in vehicle_count) == 1

    model.served = pyo.Constraint(customer_count, rule=served)

    return model

### Method Call MIP solver
def call_MIP(df_customers, selected_vehicles, vehicle_capacity, M, subtour):
    customer_indexes = list(df_customers.index.map(str))
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

def call_MIP_reorganize(df_customers, selected_vehicles, vehicle_capacity):
    customer_indexes = list(df_customers.index.map(str))
    vehicle_indexes = [str(x) for x in selected_vehicles]

    demand_customers = {b: df_customers['demand'].loc[int(b)] for b in customer_indexes}

    FL_model = create_MIP_reorganizeModel(vehicle_indexes, customer_indexes, demand_customers, vehicle_capacity)

    solver = pyo.SolverFactory('glpk')

    res = solver.solve(FL_model, timelimit=300)

    return FL_model


### Method Trivial solution
def trivial_solution(customers, vehicle_count, vehicle_capacity, customer_count):
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

### Method Compute Objective
def computeObjective(vehicle_tours, vehicle_count, M):
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 2:
            for i in range(0, len(vehicle_tour)-1):
                obj += M[int(vehicle_tour[i]),int(vehicle_tour[i+1])]
    return obj


### Initialize objects and read Data
Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])
Location = namedtuple("Locations", ['x', 'y'])

input_location = '././data/vrp_421_41_1'
with open(input_location, 'r') as input_data_file:
    input_data = input_data_file.read()

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

# Transforming list of customer into dataframe
df_customers = pd.DataFrame(customers)
df_customers['serv_Facility'] = [-1]*(customer_count)

# Calling the trivial solver
trivial_vehicle_tours = trivial_solution(customers, vehicle_count, vehicle_capacity, customer_count)

# Copying the serving vehicles in the dataframe
for index, vehicle_tour  in enumerate(trivial_vehicle_tours):
    for next_customer in vehicle_tour[1:]:
        df_customers['serv_Facility'].loc[int(next_customer)] = index
df_customers['serv_Facility'].loc[0] = -1

# Compute distance matrix
M = reduced_d_matrix(locations)

edgeMatrix = np.zeros((vehicle_count, customer_count, customer_count))
visitingMatrix = np.zeros((vehicle_count, customer_count))

for i, vehicle_tour in enumerate(trivial_vehicle_tours):
    for j in range(len(vehicle_tour)-1):
        if int(vehicle_tour[j])!=int(vehicle_tour[j+1]):
            edgeMatrix[i,int(vehicle_tour[j]),int(vehicle_tour[j+1])] = 1
        visitingMatrix[i,int(vehicle_tour[j])] = 1

draw_problem(depot, df_customers)

best_objective = computeObjective(trivial_vehicle_tours, vehicle_count, M)
best_edgeMatrix = edgeMatrix
best_visitingMatrix = visitingMatrix
best_df_customers = df_customers
best_vehicle_tours = trivial_vehicle_tours

random.seed(0)
for iter in range(100):
    print('Iter:{}'.format(iter))

    temp_selected_vehicles = random.sample(range(vehicle_count), 2)

    if (len(df_customers.loc[df_customers['serv_Facility'].isin([-1]+temp_selected_vehicles)]))>100:
        temp_selected_vehicles.remove(selected_vehicles[0])

        temp_reduced_df_customers = df_customers.loc[df_customers['serv_Facility'].isin([-1]+temp_selected_vehicles)]

        FL_model = call_MIP_reorganize(temp_reduced_df_customers, temp_selected_vehicles, vehicle_capacity)

        y = FL_model.y.extract_values()

        keys = [key for key in y.keys() if y[key] == 1]
        for key in keys:
            temp_reduced_df_customers['serv_Facility'].loc[key[1]] = key[0]

        #
        #
        # random_serv_Facility = random.choices(temp_selected_vehicles,k=len(df_customers.loc[df_customers['serv_Facility'].isin([-1]+temp_selected_vehicles)])-1)
        #
        # temp_reduced_df_customers['serv_Facility'].loc[1:] = random_serv_Facility


        selected_vehicles_list = temp_selected_vehicles

        for selected_vehicles in selected_vehicles_list:

            selected_vehicles = [selected_vehicles]
            reduced_df_customers = temp_reduced_df_customers.loc[temp_reduced_df_customers['serv_Facility'].isin([-1] + selected_vehicles)]

            subtours = []
            end = False

            while(not end):
                FL_model = call_MIP(reduced_df_customers, selected_vehicles, vehicle_capacity, M, subtours)

                y = FL_model.y.extract_values()

                keys = [key for key in y.keys() if y[key]==1]

                edgeMatrix[selected_vehicles, :, :] = 0
                visitingMatrix[selected_vehicles,:] = 0

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
                    while (next_customer<customer_count):
                        if edgeMatrix[i, current_customer, next_customer]==1:
                            df_customers['serv_Facility'].loc[next_customer] = i
                            current_customer = next_customer
                            visitingMatrix_copy[i, current_customer] = 0
                            if current_customer == 0:
                                if len(vehicle_tours[i])!=np.sum(visitingMatrix[i]):
                                    subtour = [str(s) for s in np.where(visitingMatrix_copy[i]==1)[0].tolist()]
                                    subtours.append(subtour)
                                    no_subtour[i] = False
                                break
                            vehicle_tours[i].append(str(current_customer))
                            next_customer = 0
                        else:
                            next_customer+=1
                    vehicle_tours[i].append('0')

                if no_subtour.all() == True:
                    end = True

            df_customers['serv_Facility'].loc[0] = -1

            obj = 0
            for v in range(0, vehicle_count):
                vehicle_tour = vehicle_tours[v]
                if len(vehicle_tour) > 2:
                    for i in range(0, len(vehicle_tour)-1):
                        obj += M[int(vehicle_tour[i]),int(vehicle_tour[i+1])]

    else:
        reduced_df_customers = df_customers.loc[df_customers['serv_Facility'].isin([-1] + temp_selected_vehicles)]
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
                        df_customers['serv_Facility'].loc[next_customer] = i
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

        df_customers['serv_Facility'].loc[0] = -1

        obj = 0
        for v in range(0, vehicle_count):
            vehicle_tour = vehicle_tours[v]
            if len(vehicle_tour) > 2:
                for i in range(0, len(vehicle_tour) - 1):
                    obj += M[int(vehicle_tour[i]), int(vehicle_tour[i + 1])]


    objective = computeObjective(vehicle_tours, vehicle_count, M)
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




# prepare the solution in the specified output format
outputData = '%.2f' % best_objective + ' ' + str(0) + '\n'
for v in range(0, vehicle_count):
    outputData += ' '.join([customer for customer in best_vehicle_tours[v]])+ '\n'

draw_problem(depot, best_df_customers)

print(best_df_customers)
print(outputData)

print('end')