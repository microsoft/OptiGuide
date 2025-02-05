import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class LogisticsNetworkDesign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.Number_of_Depots > 0 and self.Number_of_Cities > 0
        assert self.Min_Depot_Cost >= 0 and self.Max_Depot_Cost >= self.Min_Depot_Cost
        assert self.City_Cost_Lower_Bound >= 0 and self.City_Cost_Upper_Bound >= self.City_Cost_Lower_Bound
        assert self.Min_Depot_Capacity > 0 and self.Max_Depot_Capacity >= self.Min_Depot_Capacity

        depot_costs = np.random.randint(self.Min_Depot_Cost, self.Max_Depot_Cost + 1, self.Number_of_Depots)
        city_costs = np.random.randint(self.City_Cost_Lower_Bound, self.City_Cost_Upper_Bound + 1, (self.Number_of_Depots, self.Number_of_Cities))
        depot_capacities = np.random.randint(self.Min_Depot_Capacity, self.Max_Depot_Capacity + 1, self.Number_of_Depots)
        city_demands = np.random.randint(1, 10, self.Number_of_Cities)
        
        transport_scenarios = [{} for _ in range(self.No_of_Scenarios)]
        for s in range(self.No_of_Scenarios):
            transport_scenarios[s]['demand'] = {c: max(0, np.random.gamma(city_demands[c], city_demands[c] * self.Demand_Variation)) for c in range(self.Number_of_Cities)}

        return {
            "depot_costs": depot_costs,
            "city_costs": city_costs,
            "depot_capacities": depot_capacities,
            "city_demands": city_demands,
            "transport_scenarios": transport_scenarios
        }
        
    def solve(self, instance):
        depot_costs = instance['depot_costs']
        city_costs = instance['city_costs']
        depot_capacities = instance['depot_capacities']
        transport_scenarios = instance['transport_scenarios']
        
        model = Model("LogisticsNetworkDesign")
        number_of_depots = len(depot_costs)
        number_of_cities = len(city_costs[0])
        no_of_scenarios = len(transport_scenarios)

        # Decision variables
        depot_vars = {d: model.addVar(vtype="B", name=f"Depot_{d}") for d in range(number_of_depots)}
        city_vars = {(d, c): model.addVar(vtype="B", name=f"Depot_{d}_City_{c}") for d in range(number_of_depots) for c in range(number_of_cities)}

        # Objective: minimize the expected total cost including depot costs and city assignment costs
        model.setObjective(
            quicksum(depot_costs[d] * depot_vars[d] for d in range(number_of_depots)) +
            quicksum(city_costs[d, c] * city_vars[d, c] for d in range(number_of_depots) for c in range(number_of_cities)) +
            (1 / no_of_scenarios) * quicksum(quicksum(transport_scenarios[s]['demand'][c] * city_vars[d, c] for c in range(number_of_cities)) for d in range(number_of_depots) for s in range(no_of_scenarios)), "minimize"
        )
        
        # Constraints: Each city demand is met by exactly one depot
        for c in range(number_of_cities):
            model.addCons(quicksum(city_vars[d, c] for d in range(number_of_depots)) == 1, f"City_{c}_Demand")
        
        # Constraints: Only open depots can serve cities
        for d in range(number_of_depots):
            for c in range(number_of_cities):
                model.addCons(city_vars[d, c] <= depot_vars[d], f"Depot_{d}_Serve_{c}")

        # Constraints: Depots cannot exceed their capacity using Big M in each scenario
        for s in range(no_of_scenarios):
            for d in range(number_of_depots):
                model.addCons(quicksum(transport_scenarios[s]['demand'][c] * city_vars[d, c] for c in range(number_of_cities)) <= depot_capacities[d] * depot_vars[d], f"Depot_{d}_Scenario_{s}_Capacity")
 
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Depots': 150,
        'Number_of_Cities': 45,
        'City_Cost_Lower_Bound': 225,
        'City_Cost_Upper_Bound': 3000,
        'Min_Depot_Cost': 843,
        'Max_Depot_Cost': 5000,
        'Min_Depot_Capacity': 39,
        'Max_Depot_Capacity': 945,
        'No_of_Scenarios': 10,
        'Demand_Variation': 0.52,
    }

    logistics_network_optimizer = LogisticsNetworkDesign(parameters, seed=42)
    instance = logistics_network_optimizer.generate_instance()
    solve_status, solve_time, objective_value = logistics_network_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")