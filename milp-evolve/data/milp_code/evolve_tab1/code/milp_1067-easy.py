import random
import time
import numpy as np
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

    ################# data generation #################
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

        # Fluctuating raw material costs (over time)
        time_periods = np.random.randint(5, 15)
        raw_material_costs = np.random.randint(10, 50, size=(graph.number_of_nodes, time_periods))

        # Machine availability (binary over time)
        machine_availability = np.random.randint(0, 2, size=(graph.number_of_nodes, time_periods))

        # Labor costs (varying over time)
        labor_costs = np.random.randint(15, 60, size=time_periods)

        # Skill levels of maintenance staff (binary, skill level 0: low, 1: high)
        staff_skill_levels = np.random.randint(0, 2, size=graph.number_of_nodes)

        # Machine breakdown probabilities
        machine_breakdown_probs = np.random.uniform(0, 0.1, size=(graph.number_of_nodes, time_periods))

        # Energy consumption rates per machine
        energy_consumption_rates = np.random.uniform(1, 5, size=graph.number_of_nodes)

        # Energy cost per unit
        energy_cost_per_unit = np.random.uniform(0.1, 0.5)

        # Environmental penalty (for higher energy consumption)
        environmental_penalty = np.random.uniform(0.5, 2.0)

        res = {
            'graph': graph,
            'demands': demands,
            'capacities': capacities,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'raw_material_costs': raw_material_costs,
            'machine_availability': machine_availability,
            'labor_costs': labor_costs,
            'time_periods': time_periods,
            'staff_skill_levels': staff_skill_levels,
            'machine_breakdown_probs': machine_breakdown_probs,
            'energy_consumption_rates': energy_consumption_rates,
            'energy_cost_per_unit': energy_cost_per_unit,
            'environmental_penalty': environmental_penalty
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
        time_periods = instance['time_periods']
        staff_skill_levels = instance['staff_skill_levels']
        machine_breakdown_probs = instance['machine_breakdown_probs']
        energy_consumption_rates = instance['energy_consumption_rates']
        energy_cost_per_unit = instance['energy_cost_per_unit']
        environmental_penalty = instance['environmental_penalty']

        model = Model("CapacitatedHubLocation")

        # Add variables
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        material_usage_vars = {(i, t): model.addVar(vtype="C", name=f"material_usage_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        machine_state_vars = {(i, t): model.addVar(vtype="B", name=f"machine_state_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        waste_vars = {(i, t): model.addVar(vtype="C", name=f"waste_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        breakdown_vars = {(i, t): model.addVar(vtype="B", name=f"breakdown_{i}_{t}") for i in graph.nodes for t in range(time_periods)}
        energy_vars = {(i, t): model.addVar(vtype="C", name=f"energy_{i}_{t}") for i in graph.nodes for t in range(time_periods)}

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

        # Ensure staff skill levels match the machine requirements
        for t in range(time_periods):
            for node in graph.nodes:
                model.addCons(machine_state_vars[node, t] <= staff_skill_levels[node], name=f"SkillMatch_{node}_{t}")

        # Constraints for machine breakdown and worker safety
        for t in range(time_periods):
            for node in graph.nodes:
                model.addCons(breakdown_vars[node, t] <= machine_breakdown_probs[node, t], name=f"Breakdown_{node}_{t}")

        # Constraints for energy consumption
        for t in range(time_periods):
            for node in graph.nodes:
                model.addCons(energy_vars[node, t] == machine_state_vars[node, t] * energy_consumption_rates[node], name=f"EnergyConsumption_{node}_{t}")

        # Objective function: Minimize the total cost including labor, waste penalties, energy costs, and environmental penalties
        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        material_costs = quicksum(material_usage_vars[i, t] * raw_material_costs[i, t] for i in graph.nodes for t in range(time_periods))
        total_waste_penalty = quicksum(waste_vars[i, t] for i in graph.nodes for t in range(time_periods))
        labor_cost = quicksum(machine_state_vars[i, t] * labor_costs[t] for i in graph.nodes for t in range(time_periods))
        energy_cost = quicksum(energy_vars[i, t] * energy_cost_per_unit for i in graph.nodes for t in range(time_periods))
        environmental_impact_penalty = quicksum(energy_vars[i, t] * environmental_penalty for i in graph.nodes for t in range(time_periods))
        
        total_cost = (hub_opening_cost + connection_total_cost + material_costs + total_waste_penalty +
                    labor_cost + energy_cost + environmental_impact_penalty)

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 100,
        'edge_probability': 0.8,
        'affinity': 240,
        'graph_type': 'erdos_renyi',
    }

    hub_location_problem = CapacitatedHubLocation(parameters, seed=seed)
    instance = hub_location_problem.get_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")