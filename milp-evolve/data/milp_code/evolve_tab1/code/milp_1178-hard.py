import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DLAP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_clients_and_depots(self):
        depots = range(self.n_depots)
        clients = range(self.n_clients)

        # Distance costs
        distance_costs = np.random.randint(self.min_distance_cost, self.max_distance_cost + 1, (self.n_depots, self.n_clients))

        # Depot costs and capacities
        depot_costs = np.random.randint(self.min_depot_cost, self.max_depot_cost + 1, self.n_depots)
        depot_capacities = np.random.randint(self.min_depot_capacity, self.max_depot_capacity + 1, self.n_depots)

        # Client demands
        client_demands = np.random.randint(self.min_client_demand, self.max_client_demand + 1, self.n_clients)

        res = {
            'depots': depots, 
            'clients': clients, 
            'distance_costs': distance_costs,
            'depot_costs': depot_costs,
            'depot_capacities': depot_capacities,
            'client_demands': client_demands,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        depots = instance['depots']
        clients = instance['clients']
        distance_costs = instance['distance_costs']
        depot_costs = instance['depot_costs']
        depot_capacities = instance['depot_capacities']
        client_demands = instance['client_demands']

        model = Model("DLAP")
        
        # Variables
        assign_vars = { (i, j): model.addVar(vtype="B", name=f"assign_{i+1}_{j+1}") for i in depots for j in clients}
        depot_vars = { i: model.addVar(vtype="B", name=f"depot_{i+1}") for i in depots }
        
        # Objective
        objective_expr = quicksum(distance_costs[i, j] * assign_vars[i, j] for i in depots for j in clients)
        objective_expr += quicksum(depot_costs[i] * depot_vars[i] for i in depots)
        
        model.setObjective(objective_expr, "minimize")
        
        # Constraints
        for j in clients:
            model.addCons(quicksum(assign_vars[i, j] for i in depots) == 1, f"demand_satisfy_{j+1}")

        M = sum(client_demands)
        
        for i in depots:
            model.addCons(quicksum(assign_vars[i, j] * client_demands[j] for j in clients) <= depot_capacities[i] * depot_vars[i], f"capacity_{i+1}")
            model.addCons(quicksum(assign_vars[i, j] for j in clients) <= M * depot_vars[i], f"big_M_{i+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_depots': 20,
        'n_clients': 100,
        'min_depot_cost': 5000,
        'max_depot_cost': 20000,
        'min_distance_cost': 300,
        'max_distance_cost': 1500,
        'min_depot_capacity': 1000,
        'max_depot_capacity': 2400,
        'min_client_demand': 80,
        'max_client_demand': 300,
    }

    dlap = DLAP(parameters, seed=seed)
    instance = dlap.generate_clients_and_depots()
    solve_status, solve_time = dlap.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")