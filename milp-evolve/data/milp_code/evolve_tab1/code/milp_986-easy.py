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

class WarehouseLayoutOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_units > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_facility_space > 0 and self.max_facility_space >= self.min_facility_space
        
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_units))
        spaces = np.random.randint(self.min_facility_space, self.max_facility_space + 1, self.n_facilities)
        demands = np.random.randint(1, 10, self.n_units)

        graph = Graph.barabasi_albert(self.n_facilities, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(10):
            if node not in used_nodes:
                inequalities.add((node,))
        
        # Generate network flow costs
        graph_flow = nx.erdos_renyi_graph(self.n_facilities, 0.1, seed=self.seed)
        adj_matrix = nx.to_numpy_array(graph_flow)
        flow_costs = (adj_matrix * np.random.randint(1, 20, (self.n_facilities, self.n_facilities))).astype(int)
        
        return {
            "facility_costs": facility_costs,
            "transport_costs": transport_costs,
            "spaces": spaces,
            "demands": demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_weights": edge_weights,
            "flow_costs": flow_costs
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        transport_costs = instance['transport_costs']
        spaces = instance['spaces']
        demands = instance['demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_weights = instance['edge_weights']
        flow_costs = instance['flow_costs']
        
        model = Model("WarehouseLayoutOptimization")
        n_facilities = len(facility_costs)
        n_units = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.edges}
        flow_vars = {(i, j): model.addVar(vtype="I", name=f"Flow_{i}_{j}") for i in range(n_facilities) for j in range(n_facilities)}
        
        # Objective: minimize the total cost including flow costs, and unused space penalties, while maximizing load balance
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f, u] * transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)) +
            quicksum(edge_weights[i] * edge_vars[edge] for i, edge in enumerate(graph.edges)) +
            quicksum(flow_costs[i, j] * flow_vars[i, j] for i in range(n_facilities) for j in range(n_facilities)) -
            50 * quicksum(transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)), "minimize"
        )
        
        # Constraints: Each unit demand is met by exactly one facility
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) == 1, f"Unit_{u}_Demand")
        
        # Constraints: Only open facilities can serve units
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f], f"Facility_{f}_Serve_{u}")
        
        # Constraints: Facilities cannot exceed their space
        for f in range(n_facilities):
            model.addCons(quicksum(demands[u] * transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")
        
        # Constraints: Facility Graph Cliques
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(facility_vars[node] for node in group) <= 1, f"Clique_{count}")

        # Flow conservation constraints
        for i in range(n_facilities):
            model.addCons(
                quicksum(flow_vars[i, j] for j in range(n_facilities) if i != j) ==
                quicksum(flow_vars[j, i] for j in range(n_facilities) if i != j),
                f"Flow_Conservation_{i}"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 37,
        'n_units': 112,
        'min_transport_cost': 15,
        'max_transport_cost': 1500,
        'min_facility_cost': 2250,
        'max_facility_cost': 5000,
        'min_facility_space': 67,
        'max_facility_space': 90,
        'affinity': 15,
    }
    # New parameter for randomness seed in graph
    parameters['flow_cost_range_min'] = 1
    parameters['flow_cost_range_max'] = 20

    warehouse_layout_optimizer = WarehouseLayoutOptimization(parameters, seed=42)
    instance = warehouse_layout_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_layout_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")