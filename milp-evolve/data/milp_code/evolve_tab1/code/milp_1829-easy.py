import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
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
        graph = nx.random_geometric_graph(self.n_nodes, radius=self.radius)
        pos = nx.get_node_attributes(graph, 'pos')
        customers = random.sample(list(graph.nodes), self.n_customers)
        facilities = random.sample(list(graph.nodes), self.n_facilities)
        
        demands = np.random.poisson(self.avg_demand, self.n_customers)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.randint(self.n_facilities, self.fixed_cost_interval)
        transportation_costs = self.generate_transportation_costs(graph, pos, customers, facilities)

        single_vehicle_capacity = self.randint(1, self.vehicle_capacity_interval)[0]
        single_vehicle_cost_per_km = np.random.uniform(self.vehicle_cost_min, self.vehicle_cost_max)

        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'single_vehicle_capacity': single_vehicle_capacity,
            'single_vehicle_cost_per_km': single_vehicle_cost_per_km
        }

        return res

    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        single_vehicle_capacity = instance['single_vehicle_capacity']
        single_vehicle_cost_per_km = instance['single_vehicle_cost_per_km']

        n_customers = len(demands)
        n_facilities = len(capacities)

        model = Model("SimplifiedFacilityLocation")

        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="I", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        unmet_demand = {i: model.addVar(vtype="I", name=f"UnmetDemand_{i}") for i in range(n_customers)}

        fixed_costs_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities))
        transportation_costs_expr = quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        vehicle_costs_expr = quicksum(single_vehicle_cost_per_km * open_facilities[j] * capacities[j] / single_vehicle_capacity for j in range(n_facilities))

        unmet_demand_penalty = 1000

        objective_expr = (
            fixed_costs_expr 
            + transportation_costs_expr 
            + vehicle_costs_expr 
            + quicksum(unmet_demand_penalty * unmet_demand[i] for i in range(n_customers))
        )

        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) + unmet_demand[i] == demands[i], f"Demand_{i}")

        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 55,
        'n_facilities': 74,
        'n_nodes': 84,
        'radius': 0.38,
        'avg_demand': 735,
        'capacity_interval': (112, 1811),
        'fixed_cost_interval': (2500, 2775),
        'vehicle_capacity_interval': (375, 750),
        'vehicle_cost_min': 3.38,
        'vehicle_cost_max': 26.25,
    }

    simplified_facility_location = SimplifiedFacilityLocation(parameters, seed=seed)
    instance = simplified_facility_location.generate_instance()
    solve_status, solve_time = simplified_facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")