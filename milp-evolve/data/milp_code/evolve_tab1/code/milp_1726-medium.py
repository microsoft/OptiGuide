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
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)
        return Graph(number_of_nodes, edges, degrees, neighbors)
############# Helper function #############

class SchoolBusRouting:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def route_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_students, 1) - rand(1, self.n_buses))**2 +
            (rand(self.n_students, 1) - rand(1, self.n_buses))**2
        )
        return costs

    def generate_instance(self):
        students = self.randint(self.n_students, self.student_interval)
        capacities = self.randint(self.n_buses, self.capacity_interval)
        
        fixed_costs = (
            self.randint(self.n_buses, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_buses, self.fixed_cost_cste_interval)
        )
        route_costs = self.route_costs() * students[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(students) / np.sum(capacities)
        capacities = np.round(capacities)
        
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_buses, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_buses, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")
        
        total_fixed_cost = np.sum(fixed_costs)
        budget = self.budget_fraction * total_fixed_cost
        
        res = {
            'students': students,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'route_costs': route_costs,
            'graph': graph,
            'scenarios': self.generate_scenarios(students, num_scenarios=self.num_scenarios, demand_multiplier=self.demand_multiplier),
            'budget': budget
        }

        return res
    
    def generate_scenarios(self, students, num_scenarios, demand_multiplier):
        scenarios = []
        for _ in range(num_scenarios):
            scenario = students * (1 + np.random.uniform(-demand_multiplier, demand_multiplier, size=students.shape))
            scenarios.append(scenario)
        return scenarios
    
    def solve(self, instance):
        students = instance['students']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        route_costs = instance['route_costs']
        graph = instance['graph']
        scenarios = instance['scenarios']
        budget = instance['budget']
        
        n_students = len(students)
        n_buses = len(capacities)
        
        model = Model("SchoolBusRouting")
        
        use_buses = {j: model.addVar(vtype="B", name=f"Use_{j}") for j in range(n_buses)}
        transport = {(i, j): model.addVar(vtype="B", name=f"Transport_{i}_{j}") for i in range(n_students) for j in range(n_buses)}

        objective_expr = quicksum(fixed_costs[j] * use_buses[j] for j in range(n_buses)) + quicksum(
            route_costs[i, j] * transport[i, j] for i in range(n_students) for j in range(n_buses))
        
        total_demand_upper_bound = max(np.sum(scenario) for scenario in scenarios)
        model.addCons(quicksum(capacities[j] * use_buses[j] for j in range(n_buses)) >= total_demand_upper_bound, "TotalDemand")
        
        # New knapsack constraint to replace the independent set constraint
        model.addCons(quicksum(fixed_costs[j] * use_buses[j] for j in range(n_buses)) <= budget, "BudgetConstraint")
        
        # Symmetry breaking constraints:
        for j in range(1, n_buses):
            model.addCons(use_buses[j] <= use_buses[j - 1], f"SymmetryBreak_{j}")

        for scenario_index, scenario in enumerate(scenarios):
            for j in range(n_buses):
                model.addCons(quicksum(transport[i, j] * scenario[i] for i in range(n_students)) <= capacities[j] * use_buses[j],
                              f"Capacity_{j}_Scenario_{scenario_index}")
            for i in range(n_students):
                model.addCons(quicksum(transport[i, j] for j in range(n_buses)) >= 1, f"Demand_{i}_Scenario_{scenario_index}")

            for i in range(n_students):
                for j in range(n_buses):
                    model.addCons(transport[i, j] <= use_buses[j], f"Tightening_{i}_{j}_Scenario_{scenario_index}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_students': 240,
        'n_buses': 55,
        'student_interval': (50, 360),
        'capacity_interval': (0, 6),
        'fixed_cost_scale_interval': (500, 555),
        'fixed_cost_cste_interval': (0, 2),
        'ratio': 3.75,
        'graph_type': 'barabasi_albert',
        'edge_probability': 0.45,
        'affinity': 6,
        'num_scenarios': 7,
        'demand_multiplier': 0.24,
        'budget_fraction': 0.59,
    }

    bus_routing = SchoolBusRouting(parameters, seed=seed)
    instance = bus_routing.generate_instance()
    solve_status, solve_time = bus_routing.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")