import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

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
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

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

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class CapacitatedHubLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")

    def get_instance(self):
        graph = self.generate_graph()
        demands = np.random.randint(1, 10, size=graph.number_of_nodes)
        capacities = np.random.randint(10, 50, size=graph.number_of_nodes)
        opening_costs = np.random.randint(20, 70, size=graph.number_of_nodes)
        connection_costs = np.random.randint(1, 15, size=(graph.number_of_nodes, graph.number_of_nodes))

        time_periods = 10
        machine_availability = np.random.randint(0, 2, size=(graph.number_of_nodes, time_periods))
        energy_consumption_rates = np.random.uniform(1, 5, size=graph.number_of_nodes)
        energy_cost_per_unit = np.random.uniform(0.1, 0.5)

        ### given instance data code ends here        
        factory_capacities = np.random.uniform(100, 200, size=self.n_factories)
        factory_costs = np.random.normal(50, 10, size=self.n_factories)
        seasonal_demand_factors = 1 + 0.25 * np.sin(2 * np.pi * np.arange(graph.number_of_nodes) / graph.number_of_nodes)
        inventory_costs = np.random.normal(3, 1, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'machine_availability': machine_availability,
            'time_periods': time_periods,
            'energy_consumption_rates': energy_consumption_rates,
            'energy_cost_per_unit': energy_cost_per_unit,
            'factory_capacities': factory_capacities,
            'factory_costs': factory_costs,
            'seasonal_demand_factors': seasonal_demand_factors,
            'inventory_costs': inventory_costs,
        }
        ### new instance data code ends here
        return res

    def solve(self, instance):
        graph = instance['graph']
        demands = instance['demands']
        capacities = instance['capacities']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        machine_availability = instance['machine_availability']
        time_periods = instance['time_periods']
        energy_consumption_rates = instance['energy_consumption_rates']
        energy_cost_per_unit = instance['energy_cost_per_unit']
        
        ### given constraints and variables and objective code ends here
        factory_capacities = instance['factory_capacities']
        factory_costs = instance['factory_costs']
        seasonal_demand_factors = instance['seasonal_demand_factors']
        inventory_costs = instance['inventory_costs']

        model = Model("CapacitatedHubLocation")

        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        machine_state_vars = {(i, t): model.addVar(vtype="B", name=f"machine_state_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        energy_vars = {(i, t): model.addVar(vtype="C", name=f"energy_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        inventory_vars = {i: model.addVar(vtype="C", name=f"inventory_{i}") for i in graph.nodes}
        factory_vars = {i: model.addVar(vtype="B", name=f"factory_{i}") for i in range(self.n_factories)}

        for hub in graph.nodes:
            model.addCons(quicksum(demands[node] * routing_vars[node, hub] for node in graph.nodes) <= capacities[hub], name=f"NetworkCapacity_{hub}")

        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[node, hub] for hub in graph.nodes) == 1, name=f"ConnectionConstraints_{node}")

        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(routing_vars[node, hub] <= hub_vars[hub], name=f"ServiceProvision_{node}_{hub}")

        for t in range(time_periods):
            for node in graph.nodes:
                model.addCons(machine_state_vars[node, t] <= machine_availability[node, t], name=f"MachineAvailability_{node}_{t}")

        for t in range(time_periods):
            for node in graph.nodes:
                model.addCons(energy_vars[node, t] == machine_state_vars[node, t] * energy_consumption_rates[node], name=f"EnergyConsumption_{node}_{t}")

        for node in graph.nodes:
            model.addCons(inventory_vars[node] >= seasonal_demand_factors[node] * demands[node], name=f"SeasonalDemand_{node}")

        for i in range(self.n_factories):
            model.addCons(quicksum(demands[node] * routing_vars[node, hub] for node in graph.nodes for hub in graph.nodes if hub == i) <= factory_capacities[i] * factory_vars[i], name=f"FactoryCapacity_{i}")

        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        energy_cost = quicksum(energy_vars[i, t] * energy_cost_per_unit for i in graph.nodes for t in range(time_periods))
        inventory_cost = quicksum(inventory_costs[i] * inventory_vars[i] for i in graph.nodes)
        factory_operational_cost = quicksum(factory_costs[i] * factory_vars[i] for i in range(self.n_factories))

        total_cost = hub_opening_cost + connection_total_cost + energy_cost + inventory_cost + factory_operational_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 100,
        'edge_probability': 0.52,
        'affinity': 200,
        'graph_type': 'erdos_renyi',
        'n_factories': 2,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    hub_location_problem = CapacitatedHubLocation(parameters, seed=seed)
    instance = hub_location_problem.get_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")