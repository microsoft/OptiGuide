import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        G = nx.erdos_renyi_graph(self.n_nodes, self.density, seed=self.seed)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(self.n_nodes, self.density, seed=self.seed)
        
        # Facility costs for each node
        facility_costs = np.random.randint(1, self.max_cost, size=self.n_nodes)
        
        # Pairwise node distances (for coverage evaluation)
        node_pairs_cost = nx.adjacency_matrix(G).todense().tolist()
        
        return {
            'facility_costs': facility_costs,
            'node_pairs_cost': node_pairs_cost
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        node_pairs_cost = instance['node_pairs_cost']
        n_nodes = len(facility_costs)
        
        model = Model("FacilityLocation")
        node_decision_vars = {}

        # Create variables and set objective
        for node in range(n_nodes):
            node_decision_vars[node] = model.addVar(vtype="B", name=f"y_{node}", obj=facility_costs[node])

        # Add maximum budget constraint
        model.addCons(quicksum(node_decision_vars[node] * facility_costs[node] for node in range(n_nodes)) <= self.max_budget, "MaxBudget")

        # Add constraints for coverage
        for i in range(n_nodes):
            model.addCons(quicksum(node_decision_vars[j] for j in range(n_nodes) if node_pairs_cost[i][j] == 1) >= 1, f"NodeDegConstraint_{i}")

        # Set objective
        objective_expr = quicksum(node_decision_vars[node] * facility_costs[node] for node in range(n_nodes))
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 1750,
        'density': 0.17,
        'max_cost': 400,
        'max_budget': 1000,
    }

    facility_location_problem = FacilityLocation(parameters, seed=seed)
    instance = facility_location_problem.generate_instance()
    solve_status, solve_time = facility_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")