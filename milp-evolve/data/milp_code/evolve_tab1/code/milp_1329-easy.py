import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def efficient_greedy_clique_partition(self):
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()
        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]
        return cliques

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

class HealthcareCenterAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def gamma_dist(self, shape, scale, size):
        return np.random.gamma(shape, scale, size)

    def generate_instance(self):
        assert self.Number_of_Healthcare_Centers > 0 and self.Number_of_Zones > 0
        assert self.Min_Center_Cost >= 0 and self.Max_Center_Cost >= self.Min_Center_Cost
        assert self.Zone_Cost_Lower_Bound >= 0 and self.Zone_Cost_Upper_Bound >= self.Zone_Cost_Lower_Bound
        assert self.Min_Center_Capacity > 0 and self.Max_Center_Capacity >= self.Min_Center_Capacity
        assert self.Min_Operational_Capacity > 0 and self.Max_Center_Capacity >= self.Min_Operational_Capacity

        center_costs = self.gamma_dist(2., 200, self.Number_of_Healthcare_Centers).astype(int)
        zone_costs = self.gamma_dist(2., 100, (self.Number_of_Healthcare_Centers, self.Number_of_Zones)).astype(int)
        center_capacities = self.gamma_dist(2., 700, self.Number_of_Healthcare_Centers).astype(int)
        zone_demands = self.gamma_dist(2., 5, self.Number_of_Zones).astype(int)

        graph = Graph.barabasi_albert(self.Number_of_Healthcare_Centers, self.Affinity)
        cliques = graph.efficient_greedy_clique_partition()
        incompatibilities = set(graph.edges)

        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                incompatibilities.remove(edge)
            if len(clique) > 1:
                incompatibilities.add(clique)

        used_nodes = set()
        for group in incompatibilities:
            used_nodes.update(group)
        for node in range(self.Number_of_Healthcare_Centers):
            if node not in used_nodes:
                incompatibilities.add((node,))

        # Renewable capacities and transport scenarios
        renewable_capacity = {d: self.gamma_dist(2., 25, 1)[0] for d in range(self.Number_of_Healthcare_Centers)}
        transport_scenarios = [{} for _ in range(self.No_of_Scenarios)]
        for s in range(self.No_of_Scenarios):
            transport_scenarios[s]['demand'] = {c: np.random.normal(zone_demands[c], zone_demands[c] * self.Demand_Variation) for c in range(self.Number_of_Zones)}
        
        # New data for piecewise linear transport costs
        transport_costs = {(d, c): np.random.uniform(self.TransportCost_LB, self.TransportCost_UB) for d in range(self.Number_of_Healthcare_Centers) for c in range(self.Number_of_Zones)}

        distances = {(d, c): np.linalg.norm(np.random.uniform(0, 50, 2) - np.random.uniform(0, 50, 2).astype(int)) for d in range(self.Number_of_Healthcare_Centers) for c in range(self.Number_of_Zones)}
        transport_cost_segments = np.linspace(self.TransportCost_LB, self.TransportCost_UB, self.Segments + 1).tolist()
        transport_cost_breakpoints = [np.random.uniform(0, 50) for _ in range(self.Segments + 1)]

        return {
            "center_costs": center_costs,
            "zone_costs": zone_costs,
            "center_capacities": center_capacities,
            "zone_demands": zone_demands,
            "graph": graph,
            "incompatibilities": incompatibilities,
            "renewable_capacity": renewable_capacity,
            "transport_scenarios": transport_scenarios,
            "transport_costs": transport_costs,
            "distances": distances,
            "transport_cost_segments": transport_cost_segments,
            "transport_cost_breakpoints": transport_cost_breakpoints
        }

    def solve(self, instance):
        center_costs = instance['center_costs']
        zone_costs = instance['zone_costs']
        center_capacities = instance['center_capacities']
        zone_demands = instance['zone_demands']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        renewable_capacity = instance['renewable_capacity']
        transport_scenarios = instance['transport_scenarios']
        transport_costs = instance['transport_costs']
        distances = instance['distances']
        transport_cost_segments = instance['transport_cost_segments']
        transport_cost_breakpoints = instance['transport_cost_breakpoints']

        model = Model("HealthcareCenterAllocation")
        number_of_centers = len(center_costs)
        number_of_zones = len(zone_costs[0])

        # Decision variables
        center_vars = {d: model.addVar(vtype="B", name=f"Center_{d}") for d in range(number_of_centers)}
        zone_vars = {(d, c): model.addVar(vtype="B", name=f"Center_{d}_Zone_{c}") for d in range(number_of_centers) for c in range(number_of_zones)}
        charge_binary_vars = {d: model.addVar(vtype="B", name=f"ChargeRenewable_{d}") for d in range(number_of_centers)}
        renewable_util_vars = {d: model.addVar(vtype="C", lb=0.0, ub=renewable_capacity[d], name=f"RenewableUtil_{d}") for d in range(number_of_centers)}
        center_operation_vars = {d: model.addVar(vtype="C", lb=0, ub=center_capacities[d], name=f"CenterOperation_{d}") for d in range(number_of_centers)}

        # New Auxiliary variables and piecewise segments
        segment_vars = {(d, c, s): model.addVar(vtype="C", name=f"Segment_{d}_{c}_{s}", lb=0.0) for d in range(number_of_centers) for c in range(number_of_zones) for s in range(len(transport_cost_segments)-1)}

        # Objective: minimize the total cost including center costs, renewable utilization, variable transport costs, and zone assignment costs
        model.setObjective(
            quicksum(center_costs[d] * center_vars[d] for d in range(number_of_centers)) +
            quicksum(zone_costs[d, c] * zone_vars[d, c] for d in range(number_of_centers) for c in range(number_of_zones)) +
            quicksum(renewable_util_vars[d] for d in range(number_of_centers)) +
            quicksum(segment_vars[d, c, s] * distances[d, c] * transport_cost_segments[s] for d in range(number_of_centers) for c in range(number_of_zones) for s in range(len(transport_cost_segments)-1)),
            "minimize"
        )

        # Constraints: Each zone demand is met by exactly one center
        for c in range(number_of_zones):
            model.addCons(quicksum(zone_vars[d, c] for d in range(number_of_centers)) == 1, f"Zone_{c}_Demand")

        # Constraints: Only open centers can serve zones
        for d in range(number_of_centers):
            for c in range(number_of_zones):
                model.addCons(zone_vars[d, c] <= center_vars[d], f"Center_{d}_Serve_{c}")

        # Constraints: Centers cannot exceed their capacity
        for d in range(number_of_centers):
            model.addCons(center_operation_vars[d] <= center_capacities[d] * center_vars[d], f"Center_{d}_Capacity")

        # Constraints: Minimum operational capacity if center is open
        for d in range(number_of_centers):
            model.addCons(center_operation_vars[d] >= self.Min_Operational_Capacity * center_vars[d], f"Center_{d}_MinOperational")

        # Center Graph Incompatibilities
        for count, group in enumerate(incompatibilities):
            model.addCons(quicksum(center_vars[node] for node in group) <= 1, f"Incompatibility_{count}")

        # Renewable capacity constraints
        for d in range(number_of_centers):
            model.addCons(
                charge_binary_vars[d] * renewable_capacity[d] >= renewable_util_vars[d],
                name=f"RenewableCharging_{d}"
            )
            model.addCons(
                center_vars[d] <= charge_binary_vars[d],
                name=f"CenterCharging_{d}"
            )
            model.addCons(
                renewable_util_vars[d] <= renewable_capacity[d],
                name=f"MaxRenewableUtil_{d}"
            )

        # Scenario-based constraints
        for s in range(self.No_of_Scenarios):
            for d in range(number_of_centers):
                for c in range(number_of_zones):
                    model.addCons(
                        transport_scenarios[s]['demand'][c] * zone_vars[d, c] <= center_capacities[d],
                        name=f"Scenario_{s}_Center_{d}_Zone_{c}"
                    )

        # Piecewise Linear constraints
        for d in range(number_of_centers):
            for c in range(number_of_zones):
                model.addCons(
                    quicksum(segment_vars[d, c, s] for s in range(len(transport_cost_segments)-1)) == 1,
                    f"Piecewise_{d}_{c}_Sum"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Healthcare_Centers': 30,
        'Number_of_Zones': 168,
        'Zone_Cost_Lower_Bound': 202,
        'Zone_Cost_Upper_Bound': 3000,
        'Min_Center_Cost': 631,
        'Max_Center_Cost': 5000,
        'Min_Center_Capacity': 88,
        'Max_Center_Capacity': 283,
        'Min_Operational_Capacity': 45,
        'Affinity': 6,
        'No_of_Scenarios': 1,
        'Demand_Variation': 0.1,
        'TransportCost_LB': 225,
        'TransportCost_UB': 3000,
        'Segments': 9,
    }

    healthcare_optimizer = HealthcareCenterAllocation(parameters, seed=42)
    instance = healthcare_optimizer.generate_instance()
    solve_status, solve_time, objective_value = healthcare_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")