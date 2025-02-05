import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

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
############# Helper function #############

class EVChargingStationPlanner:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        if self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        graph = self.generate_graph()

        # Simplified inequality set based on node degrees (e.g., traffic routes, population dense areas)
        inequalities = set(graph.edges)
        for node in graph.nodes:
            neighbors = list(graph.neighbors[node])
            if len(neighbors) > 1:
                inequalities.add(tuple([node] + neighbors[:2]))

        # Random sensitive buildings to minimize EMI (Electromagnetic Interference)
        sensitive_buildings = np.random.choice(graph.nodes, self.num_sensitive_buildings, replace=False)
        emi_zones = {bld: [n for n in neighbors if np.linalg.norm(n-bld)<=self.emi_radius] for bld in sensitive_buildings}
        
        # Generate random parking capacities and air quality indices
        parking_capacity = np.random.randint(1, 10, size=self.n_nodes)
        air_quality = np.random.uniform(0, 1, size=self.n_nodes)
        
        res = {'graph': graph, 'inequalities': inequalities, 'sensitive_buildings': sensitive_buildings,
               'emi_zones': emi_zones, 'parking_capacity': parking_capacity, 'air_quality': air_quality}
        ### given instance data code ends here
        ### new instance data code ends here
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        sensitive_buildings = instance['sensitive_buildings']
        emi_zones = instance['emi_zones']
        parking_capacity = instance['parking_capacity']
        air_quality = instance['air_quality']

        model = Model("EVChargingStationPlacement")

        # Variables: station placement
        station_vars = {node: model.addVar(vtype="B", name=f"s_{node}") for node in graph.nodes}

        # Traffic and population density constraints
        for count, group in enumerate(inequalities):
            if isinstance(group, (tuple, list)) and len(group) > 1:
                model.addCons(quicksum(station_vars[node] for node in group) <= 1, name=f"constraint_{count}")

        # EMI constraints
        for building, zone in emi_zones.items():
            for node in zone:
                model.addCons(station_vars[node] == 0, name=f"emi_constraint_{building}_{node}")

        # Parking capacity constraints
        for node in graph.nodes:
            model.addCons(station_vars[node] <= parking_capacity[node], name=f"parking_constraint_{node}")

        # Objective: Maximize weighted coverage by population and air quality
        objective_expr = quicksum(station_vars[node] * (1 + air_quality[node]) for node in graph.nodes)
        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 3000,
        'affinity': 4,
        'graph_type': 'barabasi_albert',
        'num_sensitive_buildings': 100,
        'emi_radius': 18,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    ev_planner = EVChargingStationPlanner(parameters, seed=seed)
    instance = ev_planner.generate_instance()
    solve_status, solve_time = ev_planner.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")