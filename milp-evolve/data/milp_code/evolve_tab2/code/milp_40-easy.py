import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

############# Helper function #############
class Graph:
    """ Helper function: Container for a graph. """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """ Generate an Erdös-Rényi random graph with a given edge probability. """
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
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        """ Generate a Barabási-Albert random graph with a given edge probability. """
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

class DeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        P = np.random.randint(5, 30, size=(self.n_regions, self.n_members))
        region_limits = np.random.randint(50, 100, size=self.n_regions)
        benefits_weights = np.random.randint(1, 10, size=self.n_members)
        terrain_complexity = np.random.randint(1, 5, size=self.n_regions)  # New: Terrain complexity
        fuel_limits = np.random.randint(20, 70, size=self.n_regions)  # New: Fuel limits for each region
        vehicle_choices = np.random.randint(0, 2, size=self.n_members)  # Binary for vehicle type choice
        
        # New data for network flow
        flow_capacities = np.random.randint(10, 50, size=(self.n_regions, self.n_regions))
        # Ensure no self-loops
        np.fill_diagonal(flow_capacities, 0)
        
        res = {
            'P': P,
            'region_limits': region_limits,
            'benefits_weights': benefits_weights,
            'terrain_complexity': terrain_complexity,
            'fuel_limits': fuel_limits,
            'vehicle_choices': vehicle_choices,
            'flow_capacities': flow_capacities,
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        P = instance['P']
        region_limits = instance['region_limits']
        benefits_weights = instance['benefits_weights']
        terrain_complexity = instance['terrain_complexity']
        fuel_limits = instance['fuel_limits']
        vehicle_choices = instance['vehicle_choices']
        flow_capacities = instance['flow_capacities']
        
        model = Model("DeliveryOptimization")

        y = {}
        for j in range(self.n_members):
            y[j] = model.addVar(vtype='B', lb=0.0, ub=1, name="y_%s" % (j+1))
        
        # Flow variables
        flow = {}
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if i != j:  # No self-loop flows
                    flow[i, j] = model.addVar(vtype='C', lb=0.0, ub=flow_capacities[i, j], name="flow_%s_%s" % (i, j))

        # Add constraints: limits on regions
        for i in range(self.n_regions):
            model.addCons(
                quicksum(P[i, j] * y[j] for j in range(self.n_members)) <= region_limits[i], 
                "RegionLimit_%s" % (i+1)
            )
        
        # Add fuel constraints
        for i in range(self.n_regions):
            model.addCons(
                quicksum(vehicle_choices[j] * terrain_complexity[i] * y[j] for j in range(self.n_members)) <= fuel_limits[i],
                "FuelLimit_%s" % (i+1)
            )

        # Add flow capacity constraints
        for i in range(self.n_regions):
            for j in range(self.n_regions):
                if i != j:
                    model.addCons(flow[i, j] <= flow_capacities[i, j], "FlowCapacity_%s_%s" % (i, j))

        # Flow conservation for regions
        for i in range(self.n_regions):
            inflow = quicksum(flow[j, i] for j in range(self.n_regions) if j != i)
            outflow = quicksum(flow[i, j] for j in range(self.n_regions) if j != i)
            model.addCons(inflow == outflow, "FlowConservation_%s" % (i+1))

        # Objective: Minimize cost by adding an additional term to the objective function
        vehicle_costs = [random.randint(1, 3) for _ in range(self.n_members)]
        objective_expr = quicksum((benefits_weights[j] - vehicle_costs[j]) * y[j] for j in range(self.n_members))

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_regions': 100,
        'n_members': 2700,
        'graph_type': 'erdos_renyi',
        'edge_probability': 0.52,
    }

    delivery_optimization = DeliveryOptimization(parameters, seed=seed)
    instance = delivery_optimization.generate_instance()
    solve_status, solve_time = delivery_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")