import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum

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
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class ComplexHubLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        demands = np.random.randint(1, 10, size=graph.number_of_nodes)
        capacities = np.random.randint(10, 50, size=graph.number_of_nodes)
        opening_costs = np.random.randint(20, 70, size=graph.number_of_nodes)
        connection_costs = np.random.randint(1, 15, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Fluctuating raw material costs (over time)
        time_periods = 10
        raw_material_costs = np.random.randint(10, 50, size=(graph.number_of_nodes, time_periods))

        # Machine availability (binary over time)
        machine_availability = np.random.randint(0, 2, size=(graph.number_of_nodes, time_periods))

        # Labor costs (varying over time)
        labor_costs = np.random.randint(15, 60, size=time_periods)

        # Introducing diversity in transportation costs
        transportation_costs = np.random.randint(5, 20, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Introducing time-based inventory holding costs
        inventory_holding_costs = np.random.randint(1, 5, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'raw_material_costs': raw_material_costs,
            'machine_availability': machine_availability,
            'labor_costs': labor_costs,
            'transportation_costs': transportation_costs,
            'inventory_holding_costs': inventory_holding_costs,
            'time_periods': time_periods
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        raw_material_costs = instance['raw_material_costs']
        machine_availability = instance['machine_availability']
        labor_costs = instance['labor_costs']
        transportation_costs = instance['transportation_costs']
        inventory_holding_costs = instance['inventory_holding_costs']
        time_periods = instance['time_periods']

        model = Model("ComplexHubLocation")

        # Add variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        material_usage_vars = {(i, t): model.addVar(vtype="C", name=f"material_usage_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        machine_state_vars = {(i, t): model.addVar(vtype="B", name=f"machine_state_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        waste_vars = {(i, t): model.addVar(vtype="C", name=f"waste_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        inventory_vars = {(i, t): model.addVar(vtype="C", name=f"inventory_{i}_{t}") for i in graph.nodes for t in range(time_periods)}

        # Capacity Constraints
        for hub in graph.nodes:
            model.addCons(quicksum(demands[node] * routing_vars[node, hub] for node in graph.nodes) <= capacities[hub], name=f"NetworkCapacity_{hub}")

        # Connection Constraints
        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[node, hub] for hub in graph.nodes) == 1, name=f"ConnectionConstraints_{node}")

        # Ensure that routing is to an opened hub
        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(routing_vars[node, hub] <= hub_vars[hub], name=f"ServiceProvision_{node}_{hub}")

        # Ensure machine availability
        for t in range(time_periods):
            for node in graph.nodes:
                model.addCons(machine_state_vars[node, t] <= machine_availability[node, t], name=f"MachineAvailability_{node}_{t}")

        # Constraints for material usage and calculating waste
        for t in range(time_periods):
            for node in graph.nodes:
                model.addCons(material_usage_vars[node, t] <= demands[node], name=f"MaterialUsage_{node}_{t}")
                model.addCons(waste_vars[node, t] >= demands[node] - material_usage_vars[node, t], name=f"Waste_{node}_{t}")

        # New constraint for inventory holding
        for t in range(time_periods):
            for node in graph.nodes:
                # Inventory cannot exceed certain capacity
                model.addCons(inventory_vars[node, t] <= capacities[node], name=f"InventoryCapacity_{node}_{t}")

        # Objective function: Minimize the total cost including labor and waste penalties
        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        material_costs = quicksum(material_usage_vars[i, t] * raw_material_costs[i, t] for i in graph.nodes for t in range(time_periods))
        total_waste_penalty = quicksum(waste_vars[i, t] for i in graph.nodes for t in range(time_periods))
        labor_cost = quicksum(machine_state_vars[i, t] * labor_costs[t] for i in graph.nodes for t in range(time_periods))
        transportation_total_cost = quicksum(routing_vars[i, j] * transportation_costs[i, j] for i in graph.nodes for j in graph.nodes)
        inventory_holding_total_cost = quicksum(inventory_vars[i, t] * inventory_holding_costs[i] for i in graph.nodes for t in range(time_periods))

        total_cost = (hub_opening_cost + connection_total_cost + material_costs + total_waste_penalty + 
                      labor_cost + transportation_total_cost + inventory_holding_total_cost)

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 75,
        'edge_probability': 0.45,
        'affinity': 100,
        'graph_type': 'erdos_renyi',
    }

    hub_location_problem = ComplexHubLocation(parameters, seed=seed)
    instance = hub_location_problem.generate_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")