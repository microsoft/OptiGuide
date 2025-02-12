import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class OptimizedFleetManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_instance(self):
        # Vehicle Fleet data generation
        nnzrs = int(self.fleet_size * self.route_size * self.density)

        indices = np.random.choice(self.route_size, size=nnzrs)
        indices[:2 * self.route_size] = np.repeat(np.arange(self.route_size), 2)
        _, route_nfleets = np.unique(indices, return_counts=True)

        indices[:self.fleet_size] = np.random.permutation(self.fleet_size)
        i = 0
        indptr = [0]
        for n in route_nfleets:
            if i >= self.fleet_size:
                indices[i:i+n] = np.random.choice(self.fleet_size, size=n, replace=False)
            elif i + n > self.fleet_size:
                remaining_fleets = np.setdiff1d(np.arange(self.fleet_size), indices[i:self.fleet_size], assume_unique=True)
                indices[self.fleet_size:i+n] = np.random.choice(remaining_fleets, size=i+n-self.fleet_size, replace=False)
            i += n
            indptr.append(i)

        vehicle_cost = np.random.randint(self.max_vehicle_cost, size=self.route_size) + 1
        route_matrix = scipy.sparse.csc_matrix(
                (np.ones(len(indices), dtype=int), indices, indptr),
                shape=(self.fleet_size, self.route_size)).tocsr()
        indices_csr = route_matrix.indices
        indptr_csr = route_matrix.indptr

        # Additional Data for Utility Constraints from Fleet Management
        G = self.generate_random_graph()
        self.generate_maintenance_costs(G)
        capacity_limits = self.generate_capacity_data(G)
        hazard_risk_limits = self.generate_hazard_risk_limits(G)

        return { 'vehicle_cost': vehicle_cost, 'indptr_csr': indptr_csr, 'indices_csr': indices_csr,
                 'G': G, 'capacity_limits': capacity_limits, 'hazard_risk_limits': hazard_risk_limits }

    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_maintenance_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['maintenance'] = np.random.randint(1, 100)

    def generate_capacity_data(self, G):
        return {node: np.random.randint(self.min_capacity_limit, self.max_capacity_limit) for node in G.nodes}

    def generate_hazard_risk_limits(self, G):
        return {node: np.random.randint(self.min_hazard_risk, self.max_hazard_risk) for node in G.nodes}

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        vehicle_cost = instance['vehicle_cost']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        G = instance['G']
        capacity_limits = instance['capacity_limits']
        hazard_risk_limits = instance['hazard_risk_limits']

        model = Model("OptimizedFleetManagement")
        var_names = {}
        vehicle_vars = {node: model.addVar(vtype="B", name=f"x_{node}") for node in G.nodes}

        # Fleet Management Variables and Objective
        for j in range(self.route_size):
            var_names[j] = model.addVar(vtype="B", name=f"x_fleet_{j}", obj=vehicle_cost[j])

        # Add route coverage constraints
        for row in range(self.fleet_size):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"route_{row}")

        # Utility Variables and Objective
        customer_satisfaction = {node: model.addVar(vtype="C", name=f"CustSat_{node}", lb=0) for node in G.nodes}

        utility_expr = quicksum(G.nodes[node]['maintenance'] * vehicle_vars[node] for node in G.nodes)

        # Constraints for Utility
        for node in G.nodes:
            model.addCons(customer_satisfaction[node] <= capacity_limits[node], name=f"Capacity_Limit_{node}")

        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            model.addCons(quicksum(customer_satisfaction[neighbor] for neighbor in neighbors) <= hazard_risk_limits[node], name=f"Hazard_Risk_{node}")

        # Combined Objective
        objective_expr = quicksum(var_names[j] * vehicle_cost[j] for j in range(self.route_size)) + utility_expr

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'fleet_size': 750,
        'route_size': 3000,
        'density': 0.1,
        'max_vehicle_cost': 100,
        'min_n': 9,
        'max_n': 876,
        'ba_m': 150,
        'min_capacity_limit': 2250,
        'max_capacity_limit': 3000,
        'min_hazard_risk': 112,
        'max_hazard_risk': 700,
    }

    optimized_fleet_management = OptimizedFleetManagement(parameters, seed=seed)
    instance = optimized_fleet_management.generate_instance()
    solve_status, solve_time = optimized_fleet_management.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")