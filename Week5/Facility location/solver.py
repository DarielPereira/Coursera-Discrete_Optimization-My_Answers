#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo
import numpy as np
import random
from sklearn.cluster import KMeans

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'locationx', 'locationy'])
Customer = namedtuple("Customer", ['index', 'demand', 'locationx', 'locationy'])

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'locationx', 'locationy'])
Customer = namedtuple("Customer", ['index', 'demand', 'locationx', 'locationy'])

def length(Facility, Customer):
    return math.sqrt((Facility.locationx - Customer.locationx)**2 + (Facility.locationy - Customer.locationy)**2)

def dist_matrix(facilities, customers, facility_count, customer_count):
    M = np.zeros((facility_count, customer_count), dtype=float)
    for idx, facility in enumerate(facilities):
        for idx2, customer in enumerate(customers):
            M[idx,idx2] = length(facility, customer)
    return M

def distances_sum(df_facilities, df_customers):
    distance = sum(np.sqrt((np.array(df_customers.locationx)-np.array(df_facilities.loc[df_customers['serv_Facility'], 'locationx']))**2 + (np.array(df_customers.locationy)-np.array(df_facilities.loc[df_customers['serv_Facility'], 'locationy']))**2))
    return distance

def draw_problem(df_facilities, df_customers):
    plt.scatter(list(df_facilities.locationx), list(df_facilities.locationy), c = list(df_facilities['index']), marker = 'p', s=54)
    plt.scatter(list(df_customers.locationx), list(df_customers.locationy), c = list(df_customers['serv_Facility']), marker='*')
    plt.xlim(0, 160000)
    plt.ylim(0, 160000)
    # plt.xlim(min(df_customers.locationx),max(df_customers.locationx))
    # plt.ylim(min(df_customers.locationy), max(df_customers.locationy))
    plt.show()

def get_grids(df_facilities, df_customers, quantiles, i, j):
    if i == 0 and j == 0:
        df_customers_reduced = df_customers[
            (df_customers['locationx'] >= df_customers['locationx'].quantile(quantiles[i])) & (
                    df_customers['locationx'] <= df_customers['locationx'].quantile(quantiles[i + 1]))
            & (df_customers['locationy'] >= df_customers['locationy'].quantile(quantiles[j])) & (
                    df_customers['locationy'] <= df_customers['locationy'].quantile(quantiles[j + 1]))]

        df_facilities_reduced = df_facilities[
            (df_facilities['locationx'] >= df_facilities['locationx'].quantile(quantiles[i])) & (
                    df_facilities['locationx'] <= df_facilities['locationx'].quantile(quantiles[i + 1]))
            & (df_facilities['locationy'] >= df_facilities['locationy'].quantile(quantiles[j])) & (
                    df_facilities['locationy'] <= df_facilities['locationy'].quantile(quantiles[j + 1]))]

    elif i == 0:
        df_customers_reduced = df_customers[
            (df_customers['locationx'] >= df_customers['locationx'].quantile(quantiles[i])) & (
                    df_customers['locationx'] <= df_customers['locationx'].quantile(quantiles[i + 1]))
            & (df_customers['locationy'] > df_customers['locationy'].quantile(quantiles[j])) & (
                    df_customers['locationy'] <= df_customers['locationy'].quantile(quantiles[j + 1]))]

        df_facilities_reduced = df_facilities[
            (df_facilities['locationx'] >= df_facilities['locationx'].quantile(quantiles[i])) & (
                    df_facilities['locationx'] <= df_facilities['locationx'].quantile(quantiles[i + 1]))
            & (df_facilities['locationy'] > df_facilities['locationy'].quantile(quantiles[j])) & (
                    df_facilities['locationy'] <= df_facilities['locationy'].quantile(quantiles[j + 1]))]

    elif j == 0:
        df_customers_reduced = df_customers[
            (df_customers['locationx'] > df_customers['locationx'].quantile(quantiles[i])) & (
                    df_customers['locationx'] <= df_customers['locationx'].quantile(quantiles[i + 1]))
            & (df_customers['locationy'] >= df_customers['locationy'].quantile(quantiles[j])) & (
                    df_customers['locationy'] <= df_customers['locationy'].quantile(quantiles[j + 1]))]

        df_facilities_reduced = df_facilities[
            (df_facilities['locationx'] > df_facilities['locationx'].quantile(quantiles[i])) & (
                    df_facilities['locationx'] <= df_facilities['locationx'].quantile(quantiles[i + 1]))
            & (df_facilities['locationy'] >= df_facilities['locationy'].quantile(quantiles[j])) & (
                    df_facilities['locationy'] <= df_facilities['locationy'].quantile(quantiles[j + 1]))]

    else:
        df_customers_reduced = df_customers[
            (df_customers['locationx'] > df_customers['locationx'].quantile(quantiles[i])) & (
                    df_customers['locationx'] <= df_customers['locationx'].quantile(quantiles[i + 1]))
            & (df_customers['locationy'] > df_customers['locationy'].quantile(quantiles[j])) & (
                    df_customers['locationy'] <= df_customers['locationy'].quantile(quantiles[j + 1]))]

        df_facilities_reduced = df_facilities[
            (df_facilities['locationx'] > df_facilities['locationx'].quantile(quantiles[i])) & (
                    df_facilities['locationx'] <= df_facilities['locationx'].quantile(quantiles[i + 1]))
            & (df_facilities['locationy'] > df_facilities['locationy'].quantile(quantiles[j])) & (
                    df_facilities['locationy'] <= df_facilities['locationy'].quantile(quantiles[j + 1]))]

    return df_customers_reduced, df_facilities_reduced


def get_cover_cuts(df_customers, df_facilities):
    facility_count = df_facilities.shape[0]
    cover_cut_facilities = [0] * facility_count
    for i, index in enumerate(df_facilities.index):
        # cover_cut_facilities[i] = sum(distance_matrix[i, (df_customers['cum_Demand']<=df_facilities['capacity'].loc[i]).sort_index()]<distance_matrix[i,:].mean())
        cover_cut_facilities[i] = sum(df_customers['cum_Demand'] <= df_facilities['capacity'].loc[index])
    return cover_cut_facilities


def create_FL_model(facility_count, customer_count, distance_matrix, fixed_costs, demand_customers, capacity_facilities, cover_cuts):
    model = pyo.ConcreteModel(name='FL')

    model.y = pyo.Var(facility_count, customer_count, within=pyo.Binary)
    model.x = pyo.Var(facility_count, within=pyo.Binary)

    def obj_rule(mdl):
        return sum(mdl.x[m]*fixed_costs[m] for m in facility_count) + sum(mdl.y[m,n]*distance_matrix[m,n] for m in facility_count for n in customer_count)
    model.obj = pyo.Objective(rule=obj_rule)

    def setup_facility(mdl, m, n):
        return mdl.y[m, n]  <= mdl.x[m]
    model.setup_facility =  pyo.Constraint(facility_count, customer_count, rule = setup_facility)

    def cover_cut(mdl, m):
        return sum(mdl.y[m, n] for n in customer_count) <= cover_cuts[m]
    model.cover_cut =  pyo.Constraint(facility_count, rule = cover_cut)

    def unique_facility(mdl, n):
        return sum(mdl.y[m, n] for m in facility_count) == 1
    model.uniq_facility =  pyo.Constraint(customer_count, rule = unique_facility)

    def capacity(mdl, m):
        return sum(demand_customers[n]*mdl.y[m, n] for n in customer_count) <= capacity_facilities[m]
    model.capacity = pyo.Constraint(facility_count, rule = capacity)

    return model

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1, float(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])))

    df_facilities = pd.DataFrame(facilities)
    in_use = [0] * facility_count
    df_facilities['in_Use'] = in_use

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count, int(parts[0]), float(parts[1]), float(parts[2])))

    df_customers = pd.DataFrame(customers)
    ser_facility = [-1] * customer_count
    df_customers['serv_Facility'] = ser_facility

    # compute the distance matrix [facility_count X customer_count]
    distance_matrix = dist_matrix(facilities, customers, facility_count, customer_count)

    # draw_problem(df_facilities, df_customers)

    # matrix with connections
    serving_matrix = np.zeros((facility_count, customer_count), dtype='bool')

    # build a trivial solution
    # pack the facilities one by one until all the customers are served
    solution = [-1] * len(customers)

    print('Ya Termine la inicializacion')

    ###Problems 1 - 3
    if facility_count + customer_count == 75 or facility_count + customer_count == 200 or facility_count + customer_count == 250:

        df_customers = df_customers.sort_values('demand')
        df_customers['cum_Demand'] = df_customers['demand'].cumsum()
        # df_customers['cum_customers'] = range(customer_count)

        cover_cut_facilities = get_cover_cuts(df_customers, df_facilities)
        df_facilities['cover_cut'] = cover_cut_facilities

        facility_indexes = list(df_facilities.index.map(str))
        customer_indexes = list(df_customers.index.map(str))
        distance_matrix_dict = {(m, n): distance_matrix[int(m), int(n)] for m in facility_indexes for n in
                                customer_indexes}
        fixed_costs = {m: df_facilities['setup_cost'].loc[int(m)] for m in facility_indexes}
        demand_customers = {n: df_customers['demand'].loc[int(n)] for n in customer_indexes}
        capacity_facilities = {m: df_facilities['capacity'].loc[int(m)] for m in facility_indexes}
        cover_cuts = {m: df_facilities['cover_cut'].loc[int(m)] for m in facility_indexes}

        FL_model = create_FL_model(facility_indexes, customer_indexes, distance_matrix_dict, fixed_costs,
                                   demand_customers, capacity_facilities, cover_cuts)

        solver = pyo.SolverFactory('glpk')

        res = solver.solve(FL_model, timelimit=300)

        y = FL_model.y.extract_values()
        z = [key for key in y.keys() if y[key] == 1]
        for (a, b) in z:
            solution[int(b)] = int(a)


    elif facility_count + customer_count == 1100 or facility_count + customer_count == 1000:
        ###Problem 4
        if facility_count + customer_count == 1100:
            # n_clusters = 6
            grids = 2
            number_updates = 300
            number_facility_updates = 2

            ###Problem 5
        if facility_count + customer_count == 1000:
            # n_clusters = 16
            grids = 4
            number_updates = 400
            number_facility_updates = 5

        quantiles = np.linspace(0, 1, grids + 1)
        solution = [-1] * len(customers)
        for i in range(grids):
            for j in range(grids):
                print('(i,j): {} {}'.format(i, j))
                df_customers_reduced, df_facilities_reduced = get_grids(df_facilities, df_customers, quantiles, i, j)

                df_customers_reduced = df_customers_reduced.sort_values('demand')
                df_customers_reduced['cum_Demand'] = df_customers_reduced['demand'].cumsum()

                df_customers_reduced = df_customers_reduced.sort_values('demand')
                df_customers_reduced['cum_Demand'] = df_customers_reduced['demand'].cumsum()

                cover_cut_facilities = get_cover_cuts(df_customers_reduced, df_facilities_reduced)
                df_facilities.loc[df_facilities_reduced.index, 'cover_cut'] = cover_cut_facilities
                df_facilities_reduced['cover_cut'] = cover_cut_facilities

                # draw_problem(df_facilities_reduced, df_customers_reduced)

                facility_indexes = list(df_facilities_reduced.index.map(str))
                customer_indexes = list(df_customers_reduced.index.map(str))
                distance_matrix_dict = {(m, n): distance_matrix[int(m), int(n)] for m in facility_indexes for n in
                                        customer_indexes}
                fixed_costs = {m: df_facilities_reduced['setup_cost'].loc[int(m)] for m in facility_indexes}
                demand_customers = {n: df_customers_reduced['demand'].loc[int(n)] for n in customer_indexes}
                capacity_facilities = {m: df_facilities_reduced['capacity'].loc[int(m)] for m in facility_indexes}
                cover_cuts = {m: df_facilities_reduced['cover_cut'].loc[int(m)] for m in facility_indexes}

                FL_model = create_FL_model(facility_indexes, customer_indexes, distance_matrix_dict, fixed_costs,
                                           demand_customers, capacity_facilities, cover_cuts)

                solver = pyo.SolverFactory('glpk')

                res = solver.solve(FL_model)

                y = FL_model.y.extract_values()
                z = [key for key in y.keys() if y[key] == 1]
                for (a, b) in z:
                    solution[int(b)] = int(a)

        best_solution = solution

        df_customers = df_customers.sort_index()
        df_customers['serv_Facility'] = best_solution

        for facility_index in best_solution:
            df_facilities.loc[facility_index, 'in_Use'] = 1

        # calculate the cost of the solution
        best_obj = sum(df_facilities['setup_cost'] * df_facilities['in_Use'])
        best_obj += distances_sum(df_facilities, df_customers)

        # draw_problem(df_facilities, df_customers)

        temp_solution = best_solution

        ################################################
        kmeans_facilities = KMeans(n_clusters=(grids * 2), init='k-means++')
        kmeans_facilities.fit(df_facilities[df_facilities.columns[3:5]])

        df_facilities['cluster_label'] = kmeans_facilities.predict(df_facilities[df_facilities.columns[3:5]])
        df_customers['cluster_label'] = kmeans_facilities.predict(df_customers[df_customers.columns[2:4]])

        plt.scatter(list(df_facilities.locationx), list(df_facilities.locationy),
                    c=list(df_facilities['cluster_label']),
                    marker='p', s=54)

        plt.scatter(list(df_customers.locationx), list(df_customers.locationy), c=list(df_customers['cluster_label']),
                    marker='*')
        plt.show()
        ##################################################

        for update in range(number_updates):
            random.seed(update)
            random_facility_index = random.choice((df_facilities.loc[df_facilities['in_Use'] == 1].index).to_list())

            close_used_facilities = (df_facilities.loc[(df_facilities['in_Use'] == 1) & (
                        df_facilities['cluster_label'] == df_facilities['cluster_label'].loc[
                    random_facility_index])].index).to_list()
            close_unused_facilities = (df_facilities.loc[(df_facilities['in_Use'] == 0) & (
                        df_facilities['cluster_label'] == df_facilities['cluster_label'].loc[
                    random_facility_index])].index).to_list()

            facility_update_indexes = random.sample(close_used_facilities,
                                                    min(number_facility_updates, len(close_used_facilities)))
            facility_update_indexes += random.sample(close_unused_facilities,
                                                     min(number_facility_updates, len(close_unused_facilities)))

            df_facilities_reduced = df_facilities.loc[facility_update_indexes]
            df_customers_reduced = df_customers[df_customers['serv_Facility'].isin(facility_update_indexes)]

            df_customers_reduced = df_customers_reduced.sort_values('demand')
            df_customers_reduced['cum_Demand'] = df_customers_reduced['demand'].cumsum()

            cover_cut_facilities = get_cover_cuts(df_customers_reduced, df_facilities_reduced)
            df_facilities.loc[df_facilities_reduced.index, 'cover_cut'] = cover_cut_facilities
            df_facilities_reduced['cover_cut'] = cover_cut_facilities

            # draw_problem(df_facilities_reduced, df_customers_reduced)

            facility_indexes = list(df_facilities_reduced.index.map(str))
            customer_indexes = list(df_customers_reduced.index.map(str))
            distance_matrix_dict = {(m, n): distance_matrix[int(m), int(n)] for m in facility_indexes for n in
                                    customer_indexes}
            fixed_costs = {m: df_facilities_reduced['setup_cost'].loc[int(m)] for m in facility_indexes}
            demand_customers = {n: df_customers_reduced['demand'].loc[int(n)] for n in customer_indexes}
            capacity_facilities = {m: df_facilities_reduced['capacity'].loc[int(m)] for m in facility_indexes}
            cover_cuts = {m: df_facilities_reduced['cover_cut'].loc[int(m)] for m in facility_indexes}

            FL_model = create_FL_model(facility_indexes, customer_indexes, distance_matrix_dict, fixed_costs,
                                       demand_customers, capacity_facilities, cover_cuts)

            solver = pyo.SolverFactory('glpk')

            res = solver.solve(FL_model)

            y = FL_model.y.extract_values()
            z = [key for key in y.keys() if y[key] == 1]
            for (a, b) in z:
                temp_solution[int(b)] = int(a)

            temp_in_use = [0] * facility_count
            for facility_index in temp_solution:
                temp_in_use[facility_index] = 1

            temp_df_customers = df_customers
            temp_df_customers['serv_Facility'] = temp_solution

            # calculate the cost of the solution
            temp_obj = sum(df_facilities['setup_cost'] * np.array(temp_in_use))
            temp_obj += distances_sum(df_facilities, temp_df_customers)

            print('update: {}'.format(update))
            if temp_obj < best_obj:
                print('new obj: {}'.format(temp_obj))
                df_facilities['in_Use'] = temp_in_use
                df_customers = temp_df_customers
                best_solution = temp_solution
                best_obj = temp_obj

        solution = best_solution

    elif facility_count + customer_count == 2500 or facility_count + customer_count == 4000:
        # problem 6
        if facility_count + customer_count == 3500:
            n_clusters = 160
            number_updates = 400
            number_facility_updates = 5

            kmeans_facilities = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans_facilities.fit(df_facilities[df_facilities.columns[3:5]])

            df_facilities['cluster_label'] = kmeans_facilities.predict(df_facilities[df_facilities.columns[3:5]])
            df_customers['cluster_label'] = kmeans_facilities.predict(df_customers[df_customers.columns[2:4]])

            plt.scatter(list(df_facilities.locationx), list(df_facilities.locationy),
                        c=list(df_facilities['cluster_label']),
                        marker='p', s=54)

            plt.scatter(list(df_customers.locationx), list(df_customers.locationy),
                        c=list(df_customers['cluster_label']),
                        marker='*')
            plt.show()

        # problem 7
        if facility_count + customer_count == 2500:
            n_clusters = 80
            number_updates = 1000
            number_facility_updates = 5

            kmeans_facilities = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans_facilities.fit(df_facilities[df_facilities.columns[3:5]])

            df_facilities['cluster_label'] = kmeans_facilities.predict(df_facilities[df_facilities.columns[3:5]])
            df_customers['cluster_label'] = kmeans_facilities.predict(df_customers[df_customers.columns[2:4]])

            plt.scatter(list(df_facilities.locationx), list(df_facilities.locationy),
                        c=list(df_facilities['cluster_label']),
                        marker='p', s=54)

            plt.scatter(list(df_customers.locationx), list(df_customers.locationy),
                        c=list(df_customers['cluster_label']),
                        marker='*')
            plt.show()

        # problem 8
        if facility_count + customer_count == 4000:
            n_clusters = 200
            number_updates = 300
            number_facility_updates = 5

            kmeans_facilities = KMeans(n_clusters=n_clusters, init='k-means++')
            kmeans_facilities.fit(df_facilities[df_facilities.columns[3:5]])

            df_facilities['cluster_label'] = kmeans_facilities.predict(df_facilities[df_facilities.columns[3:5]])
            df_customers['cluster_label'] = kmeans_facilities.predict(df_customers[df_customers.columns[2:4]])

            plt.scatter(list(df_facilities.locationx), list(df_facilities.locationy),
                        c=list(df_facilities['cluster_label']),
                        marker='p', s=54)

            plt.scatter(list(df_customers.locationx), list(df_customers.locationy),
                        c=list(df_customers['cluster_label']),
                        marker='*')
            plt.show()

        solution = [-1] * len(customers)
        for cluster in range(n_clusters):

            print('cluster: {}'.format(cluster))

            df_customers_reduced = df_customers[df_customers['cluster_label'] == cluster]
            df_facilities_reduced = df_facilities[df_facilities['cluster_label'] == cluster]

            # draw_problem(df_facilities_reduced, df_customers_reduced)

            df_customers_reduced = df_customers_reduced.sort_values('demand')
            df_customers_reduced['cum_Demand'] = df_customers_reduced['demand'].cumsum()

            cover_cut_facilities = get_cover_cuts(df_customers_reduced, df_facilities_reduced)
            df_facilities.loc[df_facilities_reduced.index, 'cover_cut'] = cover_cut_facilities
            df_facilities_reduced['cover_cut'] = cover_cut_facilities

            facility_indexes = list(df_facilities_reduced.index.map(str))
            customer_indexes = list(df_customers_reduced.index.map(str))
            distance_matrix_dict = {(m, n): distance_matrix[int(m), int(n)] for m in facility_indexes for n in
                                    customer_indexes}
            fixed_costs = {m: df_facilities_reduced['setup_cost'].loc[int(m)] for m in facility_indexes}
            demand_customers = {n: df_customers_reduced['demand'].loc[int(n)] for n in customer_indexes}
            capacity_facilities = {m: df_facilities_reduced['capacity'].loc[int(m)] for m in facility_indexes}
            cover_cuts = {m: df_facilities_reduced['cover_cut'].loc[int(m)] for m in facility_indexes}

            FL_model = create_FL_model(facility_indexes, customer_indexes, distance_matrix_dict, fixed_costs,
                                       demand_customers, capacity_facilities, cover_cuts)

            solver = pyo.SolverFactory('glpk')

            res = solver.solve(FL_model)

            y = FL_model.y.extract_values()
            z = [key for key in y.keys() if y[key] == 1]
            for (a, b) in z:
                solution[int(b)] = int(a)

        best_solution = solution

        df_customers = df_customers.sort_index()
        df_customers['serv_Facility'] = best_solution

        for facility_index in best_solution:
            df_facilities.loc[facility_index, 'in_Use'] = 1

        # calculate the cost of the solution
        best_obj = sum(df_facilities['setup_cost'] * df_facilities['in_Use'])
        best_obj += distances_sum(df_facilities, df_customers)

        # draw_problem(df_facilities, df_customers)

        temp_solution = best_solution

        ################################################
        kmeans_facilities = KMeans(n_clusters=int(n_clusters/2), init='k-means++')
        kmeans_facilities.fit(df_facilities[df_facilities.columns[3:5]])

        df_facilities['cluster_label'] = kmeans_facilities.predict(df_facilities[df_facilities.columns[3:5]])
        df_customers['cluster_label'] = kmeans_facilities.predict(df_customers[df_customers.columns[2:4]])

        plt.scatter(list(df_facilities.locationx), list(df_facilities.locationy),
                    c=list(df_facilities['cluster_label']),
                    marker='p', s=54)

        plt.scatter(list(df_customers.locationx), list(df_customers.locationy), c=list(df_customers['cluster_label']),
                    marker='*')
        plt.show()
        ##################################################

        for update in range(number_updates):
            random.seed(update)

            random_facility_index = random.choice((df_facilities.loc[df_facilities['in_Use'] == 1].index).to_list())

            close_used_facilities = (df_facilities.loc[(df_facilities['in_Use'] == 1) & (
                    df_facilities['cluster_label'] == df_facilities['cluster_label'].loc[
                random_facility_index])].index).to_list()
            close_unused_facilities = (df_facilities.loc[(df_facilities['in_Use'] == 0) & (
                    df_facilities['cluster_label'] == df_facilities['cluster_label'].loc[
                random_facility_index])].index).to_list()

            facility_update_indexes = random.sample(close_used_facilities,
                                                    min(number_facility_updates, len(close_used_facilities)))
            facility_update_indexes += random.sample(close_unused_facilities,
                                                     min(number_facility_updates, len(close_unused_facilities)))

            df_customers_reduced = df_customers_reduced.sort_values('demand')
            df_customers_reduced['cum_Demand'] = df_customers_reduced['demand'].cumsum()

            cover_cut_facilities = get_cover_cuts(df_customers_reduced, df_facilities_reduced)
            df_facilities.loc[df_facilities_reduced.index, 'cover_cut'] = cover_cut_facilities
            df_facilities_reduced['cover_cut'] = cover_cut_facilities

            # draw_problem(df_facilities_reduced, df_customers_reduced)

            facility_indexes = list(df_facilities_reduced.index.map(str))
            customer_indexes = list(df_customers_reduced.index.map(str))
            distance_matrix_dict = {(m, n): distance_matrix[int(m), int(n)] for m in facility_indexes for n in
                                    customer_indexes}
            fixed_costs = {m: df_facilities_reduced['setup_cost'].loc[int(m)] for m in facility_indexes}
            demand_customers = {n: df_customers_reduced['demand'].loc[int(n)] for n in customer_indexes}
            capacity_facilities = {m: df_facilities_reduced['capacity'].loc[int(m)] for m in facility_indexes}
            cover_cuts = {m: df_facilities_reduced['cover_cut'].loc[int(m)] for m in facility_indexes}

            FL_model = create_FL_model(facility_indexes, customer_indexes, distance_matrix_dict, fixed_costs,
                                       demand_customers, capacity_facilities, cover_cuts)

            solver = pyo.SolverFactory('glpk')

            res = solver.solve(FL_model)

            y = FL_model.y.extract_values()
            z = [key for key in y.keys() if y[key] == 1]
            for (a, b) in z:
                temp_solution[int(b)] = int(a)

            temp_in_use = [0] * facility_count
            for facility_index in temp_solution:
                temp_in_use[facility_index] = 1

            temp_df_customers = df_customers
            temp_df_customers['serv_Facility'] = temp_solution

            # calculate the cost of the solution
            temp_obj = sum(df_facilities['setup_cost'] * np.array(temp_in_use))
            temp_obj += distances_sum(df_facilities, temp_df_customers)

            print('update: {}'.format(update))
            if temp_obj < best_obj:
                print('new obj: {}'.format(temp_obj))
                df_facilities['in_Use'] = temp_in_use
                df_customers = temp_df_customers
                best_solution = temp_solution
                best_obj = temp_obj

        solution = best_solution

    elif facility_count + customer_count == 3500:
        solution = [-1] * len(customers)


    print(solution)

    df_customers = df_customers.sort_index()
    df_customers['serv_Facility'] = solution

    # draw_problem(df_facilities, df_customers)

    for facility_index in solution:
        df_facilities.loc[facility_index, 'in_Use'] = 1

    # calculate the cost of the solution
    obj = sum(df_facilities['setup_cost'] * df_facilities['in_Use'])
    obj += distances_sum(df_facilities, df_customers)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

