import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict
from itertools import combinations

############# Helper function #############
class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        return Graph(number_of_nodes, edges, degrees, neighbors)

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
        return Graph(number_of_nodes, edges, degrees, neighbors)
############# Helper function #############

class EVChargingStationPlacement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def installation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_zones, 1) - rand(1, self.n_stations))**2 +
            (rand(self.n_zones, 1) - rand(1, self.n_stations))**2
        )
        return costs

    def generate_instance(self):
        energy_demand = self.randint(self.n_zones, self.energy_demand_interval)
        capacities = self.randint(self.n_stations, self.capacity_interval)
        
        fixed_costs = (
            self.randint(self.n_stations, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_stations, self.fixed_cost_cste_interval)
        )
        installation_costs = self.installation_costs() * energy_demand[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(energy_demand) / np.sum(capacities)
        capacities = np.round(capacities)
        
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_stations, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_stations, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")
        
        total_fixed_cost = np.sum(fixed_costs)
        budget = self.budget_fraction * total_fixed_cost
        
        res = {
            'energy_demand': energy_demand,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'installation_costs': installation_costs,
            'graph': graph,
            'scenarios': self.generate_scenarios(energy_demand, num_scenarios=self.num_scenarios, demand_multiplier=self.demand_multiplier),
            'budget': budget
        }

        # New instance data for robust optimization
        res['worst_case_variation'] = self.worst_case_variation
        
        return res
    
    def generate_scenarios(self, energy_demand, num_scenarios, demand_multiplier):
        scenarios = []
        for _ in range(num_scenarios):
            scenario = energy_demand * (1 + np.random.uniform(-demand_multiplier, demand_multiplier, size=energy_demand.shape))
            scenarios.append(scenario)
        return scenarios
    
    def solve(self, instance):
        energy_demand = instance['energy_demand']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        installation_costs = instance['installation_costs']
        graph = instance['graph']
        scenarios = instance['scenarios']
        budget = instance['budget']
        worst_case_variation = instance['worst_case_variation']
        
        n_zones = len(energy_demand)
        n_stations = len(capacities)
        
        model = Model("EVChargingStationPlacement")
        
        # New binary variables defined here
        use_stations = {j: model.addVar(vtype="B", name=f"Use_{j}") for j in range(n_stations)}
        allocate = {(i, j): model.addVar(vtype="B", name=f"Allocate_{i}_{j}") for i in range(n_zones) for j in range(n_stations)}

        # Object representing worst-case scenario
        robust_value = model.addVar(vtype="C", name="RobustValue")

        objective_expr = quicksum(fixed_costs[j] * use_stations[j] for j in range(n_stations)) + \
                         quicksum(installation_costs[i, j] * allocate[i, j] for i in range(n_zones) for j in range(n_stations)) + \
                         robust_value
        
        total_demand_upper_bound = max(np.sum(scenario) for scenario in scenarios)
        model.addCons(quicksum(capacities[j] * use_stations[j] for j in range(n_stations)) >= total_demand_upper_bound, "TotalDemand")
        
        model.addCons(quicksum(fixed_costs[j] * use_stations[j] for j in range(n_stations)) <= budget, "BudgetConstraint")
        
        # Symmetry breaking constraints
        for j in range(1, n_stations):
            model.addCons(use_stations[j] <= use_stations[j - 1], f"SymmetryBreak_{j}")

        for scenario_index, scenario in enumerate(scenarios):
            for j in range(n_stations):
                model.addCons(quicksum(allocate[i, j] * scenario[i] for i in range(n_zones)) <= capacities[j] * use_stations[j],
                              f"Capacity_{j}_Scenario_{scenario_index}")
            for i in range(n_zones):
                model.addCons(quicksum(allocate[i, j] for j in range(n_stations)) >= 1, f"Demand_{i}_Scenario_{scenario_index}")

            for i in range(n_zones):
                for j in range(n_stations):
                    model.addCons(allocate[i, j] <= use_stations[j], f"Tightening_{i}_{j}_Scenario_{scenario_index}")

        # Robust optimization constraints for worst-case scenario
        for j in range(n_stations):
            model.addCons(robust_value >= quicksum(allocate[i, j] * (energy_demand[i] * (1 + worst_case_variation)) for i in range(n_zones))
                          - capacities[j] * use_stations[j])
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_zones': 120,
        'n_stations': 110,
        'energy_demand_interval': (37, 270),
        'capacity_interval': (0, 3),
        'fixed_cost_scale_interval': (1000, 1110),
        'fixed_cost_cste_interval': (0, 1),
        'ratio': 3.75,
        'graph_type': 'barabasi_albert',
        'edge_probability': 0.45,
        'affinity': 90,
        'num_scenarios': 5,
        'demand_multiplier': 0.17,
        'budget_fraction': 0.45,
        'worst_case_variation': 0.17,
    }

    ev_charging_placement = EVChargingStationPlacement(parameters, seed=seed)
    instance = ev_charging_placement.generate_instance()
    solve_status, solve_time = ev_charging_placement.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")