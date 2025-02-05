import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_evacuation_travel_time(self, graph, node_positions, victims, facilities):
        m = len(victims)
        n = len(facilities)
        travel_times = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                victim_node = victims[i]
                facility_node = facilities[j]
                path_length = nx.shortest_path_length(graph, source=victim_node, target=facility_node, weight='weight')
                travel_times[i, j] = path_length
        return travel_times

    def generate_instance(self):
        graph = nx.random_geometric_graph(self.n_nodes, radius=self.radius)
        pos = nx.get_node_attributes(graph, 'pos')
        victims = random.sample(list(graph.nodes), self.n_victims)
        facilities = random.sample(list(graph.nodes), self.n_facilities)
        
        rescue_demand = np.random.poisson(self.avg_rescue_demand, self.n_victims)
        facility_capacities = self.randint(self.n_facilities, self.capacity_interval)
        emergency_opening_cost = self.randint(self.n_facilities, self.opening_cost_interval)
        evacuation_travel_time = self.generate_evacuation_travel_time(graph, pos, victims, facilities)
        
        res = {
            'rescue_demand': rescue_demand,
            'facility_capacities': facility_capacities,
            'emergency_opening_cost': emergency_opening_cost,
            'evacuation_travel_time': evacuation_travel_time,
        }

        return res

    def solve(self, instance):
        rescue_demand = instance['rescue_demand']
        facility_capacities = instance['facility_capacities']
        emergency_opening_cost = instance['emergency_opening_cost']
        evacuation_travel_time = instance['evacuation_travel_time']

        n_victims = len(rescue_demand)
        n_facilities = len(facility_capacities)

        model = Model("EmergencyFacilityLocation")

        open_facilities = {j: model.addVar(vtype="B", name=f"EmergencyFacilityOpen_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="I", name=f"Serve_{i}_{j}") for i in range(n_victims) for j in range(n_facilities)}
        unmet_demand = {i: model.addVar(vtype="I", name=f"UnmetDemand_{i}") for i in range(n_victims)}

        opening_costs_expr = quicksum(emergency_opening_cost[j] * open_facilities[j] for j in range(n_facilities))
        travel_time_expr = quicksum(evacuation_travel_time[i, j] * serve[i, j] for i in range(n_victims) for j in range(n_facilities))

        unmet_demand_penalty = 1000

        objective_expr = (
            opening_costs_expr 
            + travel_time_expr 
            + quicksum(unmet_demand_penalty * unmet_demand[i] for i in range(n_victims))
        )

        for i in range(n_victims):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) + unmet_demand[i] == rescue_demand[i], f"Demand_{i}")
        
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] for i in range(n_victims)) <= facility_capacities[j] * open_facilities[j], f"Capacity_{j}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_victims': 55,
        'n_facilities': 74,
        'n_nodes': 84,
        'radius': 0.38,
        'avg_rescue_demand': 735,
        'capacity_interval': (112, 1811),
        'opening_cost_interval': (2500, 2775),
    }

    emergency_facility_location = EmergencyFacilityLocation(parameters, seed=seed)
    instance = emergency_facility_location.generate_instance()
    solve_status, solve_time = emergency_facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")