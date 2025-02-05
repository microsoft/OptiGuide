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

    @staticmethod
    def barabasi_albert(number_of_nodes, edges_to_attach):
        """Generate a Barabási-Albert random graph."""
        edges = set()
        neighbors = {node: set() for node in range(number_of_nodes)}
        G = nx.barabasi_albert_graph(number_of_nodes, edges_to_attach)
        degrees = np.zeros(number_of_nodes, dtype=int)
        for edge in G.edges:
            edges.add(edge)
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class ReliefOperationRouting:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.edges_to_attach)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        hub_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        transport_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50

        max_budget = np.random.randint(1000, 5000)
        min_hubs = 2
        max_hubs = 10
        hub_capacities = np.random.randint(100, self.max_capacity, size=graph.number_of_nodes)
        shortage_penalties = np.random.randint(10, 50, size=graph.number_of_nodes)

        # Logical condition data for edge existence
        edge_exists = {(i, j): (1 if (i, j) in graph.edges else 0) for i in graph.nodes for j in graph.nodes}

        # New data for carbon emissions, noise pollution, and environmental planner approvals
        carbon_emissions = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 10
        noise_pollution = np.random.randint(1, 10, size=(graph.number_of_nodes, graph.number_of_nodes))
        environmental_approval = np.random.choice([0, 1], size=(graph.number_of_nodes, graph.number_of_nodes))

        res = {
            'graph': graph,
            'demands': demands,
            'hub_costs': hub_costs,
            'transport_costs': transport_costs,
            'max_budget': max_budget,
            'min_hubs': min_hubs,
            'max_hubs': max_hubs,
            'hub_capacities': hub_capacities,
            'shortage_penalties': shortage_penalties,
            'edge_exists': edge_exists,
            'carbon_emissions': carbon_emissions,
            'noise_pollution': noise_pollution,
            'environmental_approval': environmental_approval
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        hub_costs = instance['hub_costs']
        transport_costs = instance['transport_costs']
        max_budget = instance['max_budget']
        min_hubs = instance['min_hubs']
        max_hubs = instance['max_hubs']
        hub_capacities = instance['hub_capacities']
        shortage_penalties = instance['shortage_penalties']
        edge_exists = instance['edge_exists']
        carbon_emissions = instance['carbon_emissions']
        noise_pollution = instance['noise_pollution']
        environmental_approval = instance['environmental_approval']

        model = Model("ReliefOperationRouting")

        # Add variables
        hub_vars = {node: model.addVar(vtype="B", name=f"HubSelection_{node}") for node in graph.nodes}
        route_vars = {(i, j): model.addVar(vtype="B", name=f"Route_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Variables for unmet demand penalties
        shortage_vars = {node: model.addVar(vtype="C", name=f"Shortage_{node}") for node in graph.nodes}

        # Add carbon emission and noise pollution variables
        carbon_vars = {(i, j): model.addVar(vtype="C", name=f"Carbon_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        noise_vars = {(i, j): model.addVar(vtype="C", name=f"Noise_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Number of hubs constraint
        model.addCons(quicksum(hub_vars[node] for node in graph.nodes) >= min_hubs, name="MinHubs")
        model.addCons(quicksum(hub_vars[node] for node in graph.nodes) <= max_hubs, name="MaxHubs")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(route_vars[zone, hub] for hub in graph.nodes) + shortage_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Routing from selected hubs with logical condition for edge existence
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(route_vars[i, j] <= hub_vars[j], name=f"RouteServing_{i}_{j}")
                model.addCons(route_vars[i, j] <= edge_exists[i, j], name=f"RouteExistence_{i}_{j}")

        # Capacity constraints with logical condition for resource availability
        for j in graph.nodes:
            model.addCons(quicksum(route_vars[i, j] * demands[i] for i in graph.nodes) <= hub_capacities[j], name=f"Capacity_{j}")

        # Budget constraints
        total_cost = quicksum(hub_vars[node] * hub_costs[node] for node in graph.nodes) + \
                     quicksum(route_vars[i, j] * transport_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(shortage_vars[node] * shortage_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # Objective: Minimize total cost including deliveries and shortage penalties
        objective = total_cost + quicksum(shortage_vars[node] for node in graph.nodes)

        # Carbon emissions constraints
        max_carbon_emission = 2000  # Example value, can be adjusted
        model.addCons(quicksum(route_vars[i, j] * carbon_emissions[i, j] for i in graph.nodes for j in graph.nodes) <= max_carbon_emission, name="CarbonLimit")

        # Noise pollution constraints for each route
        max_noise_level = 1000  # Example value, can be adjusted
        model.addCons(quicksum(route_vars[i, j] * noise_pollution[i, j] for i in graph.nodes for j in graph.nodes) <= max_noise_level, name="NoiseLimit")

        # Environmental approval constraints
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(route_vars[i, j] <= environmental_approval[i, j], name=f"EnvironmentalApproval_{i}_{j}")

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 90,
        'edge_probability': 0.52,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 15,
        'max_capacity': 1684,
    }

    relief_operation = ReliefOperationRouting(parameters, seed=seed)
    instance = relief_operation.generate_instance()
    solve_status, solve_time = relief_operation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")