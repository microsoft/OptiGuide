import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class ModifiedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_transportation_costs(self, graph, node_positions, customers, facilities):
        m = len(customers)
        n = len(facilities)
        costs = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                customer_node = customers[i]
                facility_node = facilities[j]
                path_length = nx.shortest_path_length(graph, source=customer_node, target=facility_node, weight='weight')
                costs[i, j] = path_length
        return costs

    def generate_instance(self):
        graph = nx.random_geometric_graph(self.n_nodes, self.radius)
        pos = nx.get_node_attributes(graph, 'pos')
        customers = random.sample(list(graph.nodes), self.n_customers)
        facilities = random.sample(list(graph.nodes), self.n_facilities)

        demands = np.random.poisson(self.avg_demand, self.n_customers)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.randint(self.n_facilities, self.fixed_cost_interval)
        transportation_costs = self.generate_transportation_costs(graph, pos, customers, facilities)

        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']

        n_customers = len(demands)
        n_facilities = len(capacities)

        model = Model("ModifiedFacilityLocation")

        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        cover = {(i, j): model.addVar(vtype="B", name=f"Cover_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        
        # New variable for unmet demand penalties
        unmet_demand_penalties = {i: model.addVar(vtype="C", name=f"UnmetDemand_{i}") for i in range(n_customers)}

        # Objective: minimize total cost including penalties for unmet demand
        fixed_costs_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities))
        transportation_costs_expr = quicksum(transportation_costs[i, j] * cover[i, j] for i in range(n_customers) for j in range(n_facilities))
        unmet_demand_penalties_expr = quicksum(self.penalty_coefficient * unmet_demand_penalties[i] for i in range(n_customers))

        objective_expr = fixed_costs_expr + transportation_costs_expr + unmet_demand_penalties_expr
        
        model.setObjective(objective_expr, "minimize")

        # Constraints: Each customer must be assigned to exactly one facility or incur unmet demand penalty
        for i in range(n_customers):
            model.addCons(quicksum(cover[i, j] for j in range(n_facilities)) + unmet_demand_penalties[i] / self.big_M == 1, f"Cover_{i}")

        # Constraints: Facility capacities using Big M method
        for j in range(n_facilities):
            model.addCons(quicksum(demands[i] * cover[i, j] for i in range(n_customers)) <= capacities[j], f"Capacity_{j}")
            model.addCons(quicksum(demands[i] * cover[i, j] for i in range(n_customers)) <= self.big_M * open_facilities[j], f"BigM_Capacity_{j}")

        # Constraints: Facilities can only cover customers if they are open
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(cover[i, j] <= open_facilities[j], f"OpenCover_{i}_{j}")

        # New constraint: Minimum customers served by any open facility
        for j in range(n_facilities):
            model.addCons(quicksum(cover[i, j] for i in range(n_customers)) >= self.min_customers_served * open_facilities[j], f"MinCustomersServed_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 111,
        'n_facilities': 37,
        'n_nodes': 112,
        'radius': 0.31,
        'avg_demand': 105,
        'capacity_interval': (150, 2415),
        'fixed_cost_interval': (2500, 2775),
        'big_M': 10000,
        'penalty_coefficient': 500,
        'min_customers_served': 5,
    }

    facility_location = ModifiedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")