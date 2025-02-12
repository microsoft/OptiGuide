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
    def erdos_renyi(number_of_nodes, edge_probability):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.
        """
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
        terrain_complexity = np.random.randint(1, 5, size=self.n_regions)
        fuel_limits = np.random.randint(20, 70, size=self.n_regions)
        vehicle_choices = np.random.randint(0, 2, size=self.n_members)
        
        # Define machine maintenance schedules
        machine_maintenance = np.random.randint(1, 5, size=self.n_machines)
        human_resource_availability = np.random.randint(10, 40, size=self.n_regions)

        res = {
            'P': P,
            'region_limits': region_limits,
            'benefits_weights': benefits_weights,
            'terrain_complexity': terrain_complexity,
            'fuel_limits': fuel_limits,
            'vehicle_choices': vehicle_choices,
            'machine_maintenance': machine_maintenance,
            'human_resource_availability': human_resource_availability
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
        machine_maintenance = instance['machine_maintenance']
        human_resource_availability = instance['human_resource_availability']
        
        model = Model("DeliveryOptimization")

        y = {}
        for j in range(self.n_members):
            y[j] = model.addVar(vtype='B', name="y_%s" % (j+1))

        downtime_slots = {}
        for m in range(self.n_machines):
            downtime_slots[m] = model.addVar(vtype='B', name="downtime_%s" % (m+1))
        
        human_allocation = {}
        for i in range(self.n_regions):
            human_allocation[i] = model.addVar(vtype='B', name="human_alloc_%s" % (i+1))
        
        # Add Region Limit Constraints
        for i in range(self.n_regions):
            model.addCons(
                quicksum(P[i, j] * y[j] for j in range(self.n_members)) <= region_limits[i], 
                "RegionLimit_%s" % (i+1)
            )

        # Add Fuel Limit Constraints
        for i in range(self.n_regions):
            model.addCons(
                quicksum(vehicle_choices[j] * terrain_complexity[i] * y[j] for j in range(self.n_members)) <= fuel_limits[i],
                "FuelLimit_%s" % (i+1)
            )

        # Add Machine Downtime Constraints
        for m in range(self.n_machines):
            model.addCons(
                downtime_slots[m] * self.downtime_hours <= machine_maintenance[m],
                "DowntimeLimit_%s" % (m+1)
            )

        # Add Human Resource Constraints
        for i in range(self.n_regions):
            model.addCons(
                quicksum(human_allocation[i] for _ in range(self.n_members)) <= human_resource_availability[i],
                "HumanResource_%s" % (i+1)
            )

        # Objective Function: include utilization maximization
        vehicle_costs = [random.randint(1, 3) for _ in range(self.n_members)]
        team_utilization = quicksum(human_allocation[i] for i in range(self.n_regions))
        objective_expr = quicksum((benefits_weights[j] - vehicle_costs[j]) * y[j] for j in range(self.n_members))
        objective_expr += team_utilization

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_regions': 100,
        'n_members': 900,
        'graph_type': 'erdos_renyi',
        'edge_probability': 0.73,
        'n_machines': 20,
        'downtime_hours': 70,
    }

    delivery_optimization = DeliveryOptimization(parameters, seed=seed)
    instance = delivery_optimization.generate_instance()
    solve_status, solve_time = delivery_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")