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

class RobustFacilityLocation:
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
            (rand(self.n_customers, 1) - rand(1, self.n_facilities)) ** 2 +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities)) ** 2
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)

        fixed_costs = (
                self.randint(self.n_facilities, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
                self.randint(self.n_facilities, self.fixed_cost_cste_interval)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)

        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_facilities, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_facilities, self.affinity)
        elif self.graph_type == 'small_world':
            graph = Graph.small_world(self.n_facilities, self.k, self.edge_probability)
        else:
            raise ValueError("Unsupported graph type.")

        delivery_times = self.randint((self.n_customers, self.n_facilities), self.delivery_time_interval)

        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'graph': graph,
            'scenarios': self.generate_scenarios(demands, num_scenarios=self.num_scenarios, demand_multiplier=self.demand_multiplier),
            'delivery_times': delivery_times
        }
        return res

    def generate_scenarios(self, demands, num_scenarios, demand_multiplier):
        scenarios = []
        for _ in range(num_scenarios):
            scenario = demands * (1 + np.random.uniform(-demand_multiplier, demand_multiplier, size=demands.shape))
            scenarios.append(scenario)
        return scenarios

    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        graph = instance['graph']
        scenarios = instance['scenarios']
        delivery_times = instance['delivery_times']

        n_customers = len(demands)
        n_facilities = len(capacities)

        model = Model("RobustFacilityLocation")

        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        capacity_expansion = {j: model.addVar(vtype="C", name=f"CapExpansion_{j}") for j in range(n_facilities)}

        if self.continuous_assignment:
            serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        else:
            serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        delay_penalty = {(i, j, s): model.addVar(vtype="C", name=f"DelayPenalty_{i}_{j}_{s}") for i in range(n_customers) for j in range(n_facilities) for s in range(len(scenarios))}

        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) + \
                         quicksum(capacity_expansion[j] * self.expansion_cost_coefficient for j in range(n_facilities)) + \
                         quicksum(self.penalty_cost * (1 - quicksum(serve[i, j] for j in range(n_facilities))) for i in range(n_customers)) + \
                         quicksum(delay_penalty[i, j, s] * self.delay_cost_coefficient for i in range(n_customers) for j in range(n_facilities) for s in range(len(scenarios)))

        total_demand_upper_bound = max(np.sum(scenario) for scenario in scenarios)
        model.addCons(quicksum((capacities[j] + capacity_expansion[j]) * open_facilities[j] for j in range(n_facilities)) >= total_demand_upper_bound, "TotalDemand")
        model.addCons(quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) <= self.budget, "BudgetConstraint")

        for edge in graph.edges:
            model.addCons(open_facilities[edge[0]] + open_facilities[edge[1]] <= 1, f"IndependentSet_{edge[0]}_{edge[1]}")

        for scenario_index, scenario in enumerate(scenarios):
            for j in range(n_facilities):
                model.addCons(quicksum(serve[i, j] * scenario[i] for i in range(n_customers)) <= (capacities[j] + capacity_expansion[j]) * open_facilities[j],
                              f"Capacity_{j}_Scenario_{scenario_index}")
            for i in range(n_customers):
                model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}_Scenario_{scenario_index}")

                for j in range(n_facilities):
                    model.addCons(serve[i, j] <= open_facilities[j], f"Tightening_{i}_{j}_Scenario_{scenario_index}")
                    delivery_time = delivery_times[i, j]
                    model.addCons(delay_penalty[i, j, scenario_index] >= self.max_allowed_time - delivery_time, f"DelayPenalty_{i}_{j}_Scenario_{scenario_index}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 37,
        'n_facilities': 75,
        'demand_interval': (350, 2520),
        'capacity_interval': (63, 1080),
        'fixed_cost_scale_interval': (1000, 1110),
        'fixed_cost_cste_interval': (0, 22),
        'ratio': 1.88,
        'continuous_assignment': 30,
        'graph_type': 'small_world',
        'edge_probability': 0.38,
        'affinity': 40,
        'num_scenarios': 20,
        'demand_multiplier': 0.66,
        'expansion_cost_coefficient': 75,
        'penalty_cost': 200,
        'budget': 250000,
        'k': 60,
        'delivery_time_interval': (10, 50),
        'delay_cost_coefficient': 1.2,
        'max_allowed_time': 30
    }

    facility_location = RobustFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")