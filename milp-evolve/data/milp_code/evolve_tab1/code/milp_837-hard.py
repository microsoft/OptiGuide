import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SCND:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_graph(self):
        G = nx.barabasi_albert_graph(n=self.n_nodes, m=self.ba_edges, seed=self.seed)
        capacities = np.random.uniform(self.cap_min, self.cap_max, size=(self.n_nodes, self.n_nodes))
        transport_costs = np.random.uniform(self.tc_min, self.tc_max, size=(self.n_nodes, self.n_nodes))
        return G, capacities, transport_costs

    def generate_demand(self):
        demands = np.random.uniform(self.demand_min, self.demand_max, size=self.n_nodes)
        return demands

    def generate_facilities(self):
        facilities = np.random.uniform(self.facility_min, self.facility_max, size=self.n_nodes)
        opening_costs = np.random.uniform(self.opening_cost_min, self.opening_cost_max, size=self.n_nodes)
        return facilities, opening_costs

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        G, capacities, transport_costs = self.generate_random_graph()
        demands = self.generate_demand()
        facilities, opening_costs = self.generate_facilities()

        res = {
            'graph': G,
            'capacities': capacities,
            'transport_costs': transport_costs,
            'demands': demands,
            'facilities': facilities,
            'opening_costs': opening_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['graph']
        capacities = instance['capacities']
        transport_costs = instance['transport_costs']
        demands = instance['demands']
        facilities = instance['facilities']
        opening_costs = instance['opening_costs']
        
        model = Model("SCND")
        Facility_Open = {i: model.addVar(vtype="B", name=f"Facility_Open_{i}") for i in range(self.n_nodes)}
        Allocation = {(i, j): model.addVar(vtype="C", name=f"Allocation_{i}_{j}") for i in range(self.n_nodes) for j in range(self.n_nodes)}

        # Objective function
        objective_expr = quicksum(
            opening_costs[i] * Facility_Open[i]
            for i in range(self.n_nodes)
        ) + quicksum(
            Allocation[i, j] * transport_costs[i, j]
            for i in range(self.n_nodes) for j in range(self.n_nodes)
        )

        model.setObjective(objective_expr, "minimize")

        # Constraints
        for i in range(self.n_nodes):
            # Facility capacity constraint
            model.addCons(
                quicksum(Allocation[i, j] for j in range(self.n_nodes)) <= facilities[i] * Facility_Open[i],
                f"Facility_Capacity_{i}"
            )
            # Demand satisfaction constraint
            model.addCons(
                quicksum(Allocation[j, i] for j in range(self.n_nodes)) == demands[i],
                f"Demand_Satisfaction_{i}"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 120,
        'max_n_nodes': 200,
        'ba_edges': 4,
        'facility_min': 2100,
        'facility_max': 1000,
        'opening_cost_min': 5000,
        'opening_cost_max': 15000,
        'cap_min': 75,
        'cap_max': 2000,
        'tc_min': 16,
        'tc_max': 800,
        'demand_min': 700,
        'demand_max': 1000,
    }

    scnd = SCND(parameters, seed=seed)
    instance = scnd.generate_instance()
    solve_status, solve_time = scnd.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")