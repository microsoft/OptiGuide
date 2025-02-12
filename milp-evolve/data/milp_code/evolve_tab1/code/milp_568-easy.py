import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from itertools import combinations


############# Helper function #############
class Graph:
    """
    Helper function: Container for a graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        """
        Generate a Barabási-Albert random graph with a given edge probability.
        """
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

############# Main MILP Class #############
class IntegratedOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        # Production and Delivery Data Generation
        n_factories = random.randint(self.min_factories, self.max_factories)
        n_locations = random.randint(self.min_locations, self.max_locations)

        production_costs = np.random.randint(1000, 5000, size=n_factories)
        delivery_costs = np.random.randint(50, 200, size=(n_factories, n_locations))
        production_time = np.random.randint(1, 10, size=n_factories)
        inventory_capacity = np.random.randint(50, 200, size=n_factories)
        delivery_time_windows = np.random.randint(1, 10, size=n_locations)

        # EV Charging Station Data Generation
        graph = Graph.barabasi_albert(self.n_nodes, self.affinity)
        inequalities = set(graph.edges)
        for node in graph.nodes:
            neighbors = list(graph.neighbors[node])
            if len(neighbors) > 1:
                inequalities.add(tuple([node] + neighbors[:2]))

        sensitive_buildings = np.random.choice(graph.nodes, self.num_sensitive_buildings, replace=False)
        emi_zones = {bld: [n for n in neighbors if np.linalg.norm(n-bld) <= self.emi_radius] for bld in sensitive_buildings}
        parking_capacity = np.random.randint(1, 10, size=self.n_nodes)
        air_quality = np.random.uniform(0, 1, size=self.n_nodes)

        res = {
            'n_factories': n_factories,
            'n_locations': n_locations,
            'production_costs': production_costs,
            'delivery_costs': delivery_costs,
            'production_time': production_time,
            'inventory_capacity': inventory_capacity,
            'delivery_time_windows': delivery_time_windows,
            'graph': graph,
            'inequalities': inequalities,
            'sensitive_buildings': sensitive_buildings,
            'emi_zones': emi_zones,
            'parking_capacity': parking_capacity,
            'air_quality': air_quality
        }
        return res

    ################# MILP Modeling #################
    def solve(self, instance):
        n_factories = instance['n_factories']
        n_locations = instance['n_locations']
        production_costs = instance['production_costs']
        delivery_costs = instance['delivery_costs']
        production_time = instance['production_time']
        inventory_capacity = instance['inventory_capacity']
        delivery_time_windows = instance['delivery_time_windows']

        graph = instance['graph']
        inequalities = instance['inequalities']
        sensitive_buildings = instance['sensitive_buildings']
        emi_zones = instance['emi_zones']
        parking_capacity = instance['parking_capacity']
        air_quality = instance['air_quality']

        model = Model("IntegratedOptimization")

        # Variables
        y = {i: model.addVar(vtype="B", name=f"y_{i}") for i in range(n_factories)}  
        x = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(n_factories) for j in range(n_locations)}
        production_status = {i: model.addVar(vtype="B", name=f"production_status_{i}") for i in range(n_factories)}
        station_vars = {node: model.addVar(vtype="B", name=f"s_{node}") for node in graph.nodes}
        extra_cost = model.addVar(vtype="B", name="extra_cost")

        # Objective Function: Minimize total production, delivery costs, and maximize coverage
        total_cost = quicksum(production_status[i] * production_costs[i] for i in range(n_factories)) + \
                     quicksum(x[i, j] * delivery_costs[i, j] for i in range(n_factories) for j in range(n_locations)) + \
                     extra_cost * 1000
        total_coverage = quicksum(station_vars[node] * (1 + air_quality[node]) for node in graph.nodes)

        # Compound Objective: Balance cost minimization and coverage maximization
        alpha = 0.5  # Balancing factor
        compound_objective = alpha * total_cost - (1 - alpha) * total_coverage
        model.setObjective(compound_objective, "minimize")

        # Constraints
        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] for i in range(n_factories)) == 1, name=f"location_coverage_{j}")
        
        for i in range(n_factories):
            for j in range(n_locations):
                model.addCons(x[i, j] <= production_status[i], name=f"factory_to_location_{i}_{j}")

        for i in range(n_factories):
            model.addCons(production_status[i] * inventory_capacity[i] >= quicksum(x[i, j] for j in range(n_locations)), name=f"production_capacity_{i}")

        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * production_time[i] for i in range(n_factories)) <= delivery_time_windows[j], name=f"delivery_time_window_{j}")

        for count, group in enumerate(inequalities):
            if isinstance(group, (tuple, list)) and len(group) > 1:
                model.addCons(quicksum(station_vars[node] for node in group) <= 1, name=f"constraint_{count}")

        for building, zone in emi_zones.items():
            for node in zone:
                model.addCons(station_vars[node] == 0, name=f"emi_constraint_{building}_{node}")

        for node in graph.nodes:
            model.addCons(station_vars[node] <= parking_capacity[node], name=f"parking_constraint_{node}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

################# Main #################
if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_factories': 50,
        'max_factories': 250,
        'min_locations': 10,
        'max_locations': 500,
        'n_nodes': 3000,
        'affinity': 4,
        'num_sensitive_buildings': 300,
        'emi_radius': 0,
    }

    optimization = IntegratedOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")