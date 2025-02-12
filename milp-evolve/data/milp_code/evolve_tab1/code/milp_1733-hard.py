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

    @staticmethod
    def small_world(number_of_nodes, k, probability):
        G = nx.watts_strogatz_graph(number_of_nodes, k, probability)
        edges = set(G.edges())
        degrees = np.array([G.degree(node) for node in range(number_of_nodes)])
        neighbors = {node: set(G.neighbors(node)) for node in range(number_of_nodes)}
        return Graph(number_of_nodes, edges, degrees, neighbors)
############# Helper function #############

class RenewableEnergyResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_regions, 1) - rand(1, self.n_plants)) ** 2 +
            (rand(self.n_regions, 1) - rand(1, self.n_plants)) ** 2
        )
        return costs

    def generate_instance(self):
        energy_demands = self.randint(self.n_regions, self.energy_demand_interval)
        capacities = self.randint(self.n_plants, self.capacity_interval)

        fixed_costs = (
            self.randint(self.n_plants, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_plants, self.fixed_cost_cste_interval)
        )
        transportation_costs = self.unit_transportation_costs() * energy_demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(energy_demands) / np.sum(capacities)
        capacities = np.round(capacities)

        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_plants, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_plants, self.affinity)
        elif self.graph_type == 'small_world':
            graph = Graph.small_world(self.n_plants, self.k, self.edge_probability)
        else:
            raise ValueError("Unsupported graph type.")

        environmental_impacts = self.randint((self.n_regions, self.n_plants), self.environmental_impact_interval)

        res = {
            'energy_demands': energy_demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'graph': graph,
            'scenarios': self.generate_scenarios(energy_demands, num_scenarios=self.num_scenarios, demand_multiplier=self.demand_multiplier),
            'environmental_impacts': environmental_impacts
        }
        return res

    def generate_scenarios(self, energy_demands, num_scenarios, demand_multiplier):
        scenarios = []
        for _ in range(num_scenarios):
            scenario = energy_demands * (1 + np.random.uniform(-demand_multiplier, demand_multiplier, size=energy_demands.shape))
            scenarios.append(scenario)
        return scenarios

    def solve(self, instance):
        energy_demands = instance['energy_demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        graph = instance['graph']
        scenarios = instance['scenarios']
        environmental_impacts = instance['environmental_impacts']

        n_regions = len(energy_demands)
        n_plants = len(capacities)

        model = Model("RenewableEnergyResourceAllocation")

        build_plants = {j: model.addVar(vtype="B", name=f"Build_{j}") for j in range(n_plants)}
        expand_capacity = {j: model.addVar(vtype="C", name=f"Expand_{j}") for j in range(n_plants)}

        if self.continuous_delivery:
            supply = {(i, j): model.addVar(vtype="C", name=f"Supply_{i}_{j}") for i in range(n_regions) for j in range(n_plants)}
        else:
            supply = {(i, j): model.addVar(vtype="B", name=f"Supply_{i}_{j}") for i in range(n_regions) for j in range(n_plants)}

        environmental_impact = {(i, j, s): model.addVar(vtype="C", name=f"EnvironmentalImpact_{i}_{j}_{s}") for i in range(n_regions) for j in range(n_plants) for s in range(len(scenarios))}

        objective_expr = quicksum(fixed_costs[j] * build_plants[j] for j in range(n_plants)) + \
                         quicksum(transportation_costs[i, j] * supply[i, j] for i in range(n_regions) for j in range(n_plants)) + \
                         quicksum(expand_capacity[j] * self.expansion_cost_coefficient for j in range(n_plants)) + \
                         quicksum(self.penalty_cost * (1 - quicksum(supply[i, j] for j in range(n_plants))) for i in range(n_regions)) + \
                         quicksum(environmental_impact[i, j, s] * self.environmental_impact_cost_coefficient for i in range(n_regions) for j in range(n_plants) for s in range(len(scenarios)))

        total_demand_upper_bound = max(np.sum(scenario) for scenario in scenarios)
        model.addCons(quicksum((capacities[j] + expand_capacity[j]) * build_plants[j] for j in range(n_plants)) >= total_demand_upper_bound, "TotalDemand")
        model.addCons(quicksum(fixed_costs[j] * build_plants[j] for j in range(n_plants)) <= self.budget, "BudgetConstraint")

        for edge in graph.edges:
            model.addCons(build_plants[edge[0]] + build_plants[edge[1]] <= 1, f"IndependentSet_{edge[0]}_{edge[1]}")

        for scenario_index, scenario in enumerate(scenarios):
            for j in range(n_plants):
                model.addCons(quicksum(supply[i, j] * scenario[i] for i in range(n_regions)) <= (capacities[j] + expand_capacity[j]) * build_plants[j],
                              f"Capacity_{j}_Scenario_{scenario_index}")
            for i in range(n_regions):
                model.addCons(quicksum(supply[i, j] for j in range(n_plants)) >= 1, f"Demand_{i}_Scenario_{scenario_index}")

                for j in range(n_plants):
                    model.addCons(supply[i, j] <= build_plants[j], f"Tightening_{i}_{j}_Scenario_{scenario_index}")
                    environmental_impact_cost = environmental_impacts[i, j]
                    model.addCons(environmental_impact[i, j, scenario_index] >= self.max_allowed_impact - environmental_impact_cost, f"EnvironmentalImpact_{i}_{j}_Scenario_{scenario_index}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_regions': 37,
        'n_plants': 75,
        'energy_demand_interval': (350, 2520),
        'capacity_interval': (63, 1080),
        'fixed_cost_scale_interval': (1000, 1110),
        'fixed_cost_cste_interval': (0, 22),
        'ratio': 1.88,
        'continuous_delivery': 30,
        'graph_type': 'small_world',
        'edge_probability': 0.38,
        'affinity': 40,
        'num_scenarios': 20,
        'demand_multiplier': 0.66,
        'expansion_cost_coefficient': 75,
        'penalty_cost': 200,
        'budget': 250000,
        'k': 60,
        'environmental_impact_interval': (10, 50),
        'environmental_impact_cost_coefficient': 1.2,
        'max_allowed_impact': 30
    }

    energy_allocation = RenewableEnergyResourceAllocation(parameters, seed=seed)
    instance = energy_allocation.generate_instance()
    solve_status, solve_time = energy_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")