import random
import time
import numpy as np
import networkx as nx
from itertools import permutations
from pyscipopt import Model, quicksum

class Graph:
    """Helper function: Container for a graph."""
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """Generate an Erdös-Rényi random graph with a given edge probability."""
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in permutations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class ElectricalResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_graph(self):
        return Graph.erdos_renyi(self.n_nodes, self.edge_probability)

    def generate_instance(self):
        graph = self.generate_graph()
        zone_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        power_supply_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))
        substation_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        transmission_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(1000, 5000)
        min_substations = 2
        max_substations = 10
        substation_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'zone_demands': zone_demands,
            'power_supply_costs': power_supply_costs,
            'substation_costs': substation_costs,
            'transmission_costs': transmission_costs,
            'max_budget': max_budget,
            'min_substations': min_substations,
            'max_substations': max_substations,
            'substation_capacities': substation_capacities,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        zone_demands = instance['zone_demands']
        power_supply_costs = instance['power_supply_costs']
        substation_costs = instance['substation_costs']
        transmission_costs = instance['transmission_costs']
        max_budget = instance['max_budget']
        min_substations = instance['min_substations']
        max_substations = instance['max_substations']
        substation_capacities = instance['substation_capacities']

        model = Model("ElectricalResourceAllocation")

        # Add variables
        substation_vars = {node: model.addVar(vtype="B", name=f"SubstationSelection_{node}") for node in graph.nodes}
        power_vars = {(i, j): model.addVar(vtype="B", name=f"PowerRouting_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Number of substations constraint
        model.addCons(quicksum(substation_vars[node] for node in graph.nodes) >= min_substations, name="MinSubstations")
        model.addCons(quicksum(substation_vars[node] for node in graph.nodes) <= max_substations, name="MaxSubstations")

        # Electricity demand satisfaction constraint
        for zone in graph.nodes:
            model.addCons(
                quicksum(power_vars[zone, substation] for substation in graph.nodes) == 1, 
                name=f"ElectricityDemand_{zone}"
            )

        # Routing from active substations
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(power_vars[i, j] <= substation_vars[j], name=f"PowerService_{i}_{j}")

        # Capacity constraints
        for j in graph.nodes:
            model.addCons(quicksum(power_vars[i, j] * zone_demands[i] for i in graph.nodes) <= substation_capacities[j], name=f"SubstationCapacity_{j}")

        # Budget constraints
        total_cost = quicksum(substation_vars[node] * substation_costs[node] for node in graph.nodes) + \
                     quicksum(power_vars[i, j] * transmission_costs[i, j] for i in graph.nodes for j in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Objective: Minimize total cost
        objective = total_cost

        # Adding Convex Hull Formulation constraints
        alpha = {(i, j): model.addVar(vtype='C', name=f'Alpha_{i}_{j}') for i in graph.nodes for j in graph.nodes}
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(alpha[i, j] <= 1.0, name=f"Alpha_upper_{i}_{j}")
                model.addCons(alpha[i, j] >= 0.0, name=f"Alpha_lower_{i}_{j}")
                model.addCons(alpha[i, j] <= power_vars[i, j], name=f"Alpha_bound_power_{i}_{j}")
        
        for i in graph.nodes:
            for j in graph.nodes:
                # Constraints that utilize the convex hull approach
                model.addCons(alpha[i, j] * max_budget >= transmission_costs[i, j] * power_vars[i, j], name=f"ConvexHull_{i}_{j}")

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 67,
        'edge_probability': 0.8,
        'graph_type': 'erdos_renyi',
        'max_capacity': 750,
    }

    electrical_resource_allocation = ElectricalResourceAllocation(parameters, seed=seed)
    instance = electrical_resource_allocation.generate_instance()
    solve_status, solve_time = electrical_resource_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")