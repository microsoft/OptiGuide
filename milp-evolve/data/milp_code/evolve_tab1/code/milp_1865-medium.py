import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class DisasterResponseOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_travel_time(self, graph, node_positions, disaster_sites, response_centers):
        m = len(disaster_sites)
        n = len(response_centers)
        travel_times = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                disaster_node = disaster_sites[i]
                response_node = response_centers[j]
                path_length = nx.shortest_path_length(graph, source=disaster_node, target=response_node, weight='weight')
                travel_times[i, j] = path_length
        return travel_times

    def generate_instance(self):
        graph = nx.random_geometric_graph(self.n_nodes, radius=self.radius)
        pos = nx.get_node_attributes(graph, 'pos')

        disaster_sites = random.sample(list(graph.nodes), self.n_disaster_sites)
        response_centers = random.sample(list(graph.nodes), self.n_response_centers)

        dynamic_demand = self.randint(self.n_disaster_sites, self.dynamic_demand_interval)
        response_center_capacities = self.randint(self.n_response_centers, self.capacity_interval)
        opening_costs = self.randint(self.n_response_centers, self.opening_cost_interval)
        travel_time = self.generate_travel_time(graph, pos, disaster_sites, response_centers)

        medical_specialists = self.randint(self.n_response_centers, self.medical_specialists_interval)
        engineering_specialists = self.randint(self.n_response_centers, self.engineering_specialists_interval)
        specialist_costs_per_km = np.random.uniform(self.specialist_cost_min, self.specialist_cost_max, self.n_response_centers)
        
        historical_severity_data = np.random.uniform(self.severity_min, self.severity_max, self.n_regions)
        disaster_severity = np.random.gamma(2.0, 2.0, self.n_disaster_sites)

        res = {
            'dynamic_demand': dynamic_demand,
            'response_center_capacities': response_center_capacities,
            'opening_costs': opening_costs,
            'travel_time': travel_time,
            'medical_specialists': medical_specialists,
            'engineering_specialists': engineering_specialists,
            'specialist_costs_per_km': specialist_costs_per_km,
            'historical_severity_data': historical_severity_data,
            'disaster_severity': disaster_severity
        }

        return res

    def solve(self, instance):
        dynamic_demand = instance['dynamic_demand']
        response_center_capacities = instance['response_center_capacities']
        opening_costs = instance['opening_costs']
        travel_time = instance['travel_time']
        medical_specialists = instance['medical_specialists']
        engineering_specialists = instance['engineering_specialists']
        specialist_costs_per_km = instance['specialist_costs_per_km']
        historical_severity_data = instance['historical_severity_data']
        disaster_severity = instance['disaster_severity']

        n_disaster_sites = len(dynamic_demand)
        n_response_centers = len(response_center_capacities)
        n_regions = len(historical_severity_data)

        model = Model("DisasterResponseOptimization")

        open_response_centers = {j: model.addVar(vtype="B", name=f"ResponseCenterOpen_{j}") for j in range(n_response_centers)}
        serve = {(i, j): model.addVar(vtype="I", name=f"Serve_{i}_{j}") for i in range(n_disaster_sites) for j in range(n_response_centers)}
        specialist_assign = {(j, t): model.addVar(vtype="C", name=f"SpecialistAssign_{j}_{t}") for j in range(n_response_centers) for t in range(2)}
        unmet_demand = {i: model.addVar(vtype="I", name=f"UnmetDemand_{i}") for i in range(n_disaster_sites)}

        opening_costs_expr = quicksum(opening_costs[j] * open_response_centers[j] for j in range(n_response_centers))
        travel_time_expr = quicksum(travel_time[i, j] * serve[i, j] for i in range(n_disaster_sites) for j in range(n_response_centers))
        specialist_costs_expr = quicksum(specialist_costs_per_km[j] * specialist_assign[j, t] for j in range(n_response_centers) for t in range(2))
        unmet_demand_penalty = 1000
        disaster_severity_penalty = 500

        objective_expr = (
            opening_costs_expr 
            + travel_time_expr 
            + specialist_costs_expr 
            + quicksum(unmet_demand_penalty * unmet_demand[i] for i in range(n_disaster_sites))
            + quicksum(disaster_severity_penalty * (disaster_severity[i] - serve[i, j]) for i in range(n_disaster_sites) for j in range(n_response_centers))
        )

        for i in range(n_disaster_sites):
            model.addCons(quicksum(serve[i, j] for j in range(n_response_centers)) + unmet_demand[i] == dynamic_demand[i], f"Demand_{i}")

        for j in range(n_response_centers):
            model.addCons(quicksum(serve[i, j] for i in range(n_disaster_sites)) <= response_center_capacities[j] * open_response_centers[j], f"Capacity_{j}")

        for j in range(n_response_centers):
            model.addCons(specialist_assign[j, 0] * medical_specialists[j] + specialist_assign[j, 1] * engineering_specialists[j] >= response_center_capacities[j], f"SpecialistCapacity_{j}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_disaster_sites': 120,
        'n_response_centers': 75,
        'n_nodes': 300,
        'radius': 0.25,
        'dynamic_demand_interval': (100, 500),
        'capacity_interval': (750, 2250),
        'opening_cost_interval': (1000, 5000),
        'medical_specialists_interval': (90, 450),
        'engineering_specialists_interval': (100, 500),
        'specialist_cost_min': 100.0,
        'specialist_cost_max': 400.0,
        'severity_min': 7.0,
        'severity_max': 80.0,
        'n_regions': 40,
    }

    disaster_response_optimization = DisasterResponseOptimization(parameters, seed=seed)
    instance = disaster_response_optimization.generate_instance()
    solve_status, solve_time = disaster_response_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")