import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from itertools import combinations

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

    def efficient_greedy_clique_partition(self):
        """
        Partition the graph into cliques using an efficient greedy algorithm.
        """
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

class NewCityPlanner:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_city_plan(self):
        n_zones = np.random.randint(self.min_zones, self.max_zones)
        G = nx.erdos_renyi_graph(n=n_zones, p=self.er_prob, seed=self.seed)
        return G

    def generate_zone_utility_data(self, G):
        for node in G.nodes:
            G.nodes[node]['utility'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['distance'] = np.random.randint(1, 20)
    
    def generate_unwanted_routes(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                E_invalid.add(edge)
        return E_invalid

    def generate_instance(self):
        G = self.generate_random_city_plan()
        self.generate_zone_utility_data(G)
        E_invalid = self.generate_unwanted_routes(G)
        
        graph = self.convert_nx_graph_to_custom_graph(G)
        clusters = graph.efficient_greedy_clique_partition()

        return {
            'G': G,
            'E_invalid': E_invalid, 
            'clusters': clusters, 
            'capacities': np.random.randint(1, 100, len(G.nodes)),
            'transportation_costs': np.random.uniform(0.1, 1.0, len(G.nodes)),
        }
    
    def convert_nx_graph_to_custom_graph(self, G):
        edges = set(G.edges)
        degrees = np.array([degree for _, degree in G.degree()])
        neighbors = {node: set(G.neighbors(node)) for node in G.nodes}
        return Graph(len(G.nodes), edges, degrees, neighbors)

    def solve(self, instance):
        G, E_invalid, clusters = instance['G'], instance['E_invalid'], instance['clusters']
        capacities = instance['capacities']
        transportation_costs = instance['transportation_costs']
        
        model = Model("NewCityPlanner")
        zone_activation_vars = {f"z{node}":  model.addVar(vtype="B", name=f"z{node}") for node in G.nodes}
        cluster_route_usage_vars = {f"ru{u}_{v}": model.addVar(vtype="B", name=f"ru{u}_{v}") for u, v in G.edges}

        objective_expr = quicksum(
            G.nodes[node]['utility'] * zone_activation_vars[f"z{node}"]
            for node in G.nodes
        )
        
        objective_expr -= quicksum(
            G[u][v]['distance'] * cluster_route_usage_vars[f"ru{u}_{v}"]
            for u, v in E_invalid
        )
        
        objective_expr -= quicksum(
            transportation_costs[node] * zone_activation_vars[f"z{node}"]
            for node in G.nodes
        )
        
        model.setObjective(objective_expr, "maximize")

        for u, v in G.edges:
            if (u, v) in E_invalid:
                model.addCons(
                    zone_activation_vars[f"z{u}"] + zone_activation_vars[f"z{v}"] - cluster_route_usage_vars[f"ru{u}_{v}"] <= 1,
                    name=f"MallDistanceConstraints_{u}_{v}"
                )
            else:
                model.addCons(
                    zone_activation_vars[f"z{u}"] + zone_activation_vars[f"z{v}"] <= 1,
                    name=f"HousesClusterDistance_{u}_{v}"
                )

        for cl in clusters:
            for u in cl:
                for v in cl:
                    if u < v:  # Avoid repetition
                        model.addCons(
                            zone_activation_vars[f"z{u}"] + zone_activation_vars[f"z{v}"] <= 1,
                            name=f"CliqueInequality_{u}_{v}"
                        )

        for node in G.nodes:
            model.addCons(
                quicksum(zone_activation_vars[f"z{neighbor}"] for neighbor in G.neighbors(node)) <= capacities[node],
                name=f"CapacityConstraints_{node}"
            )

        mutual_exclusivity_pairs = self.generate_mutual_exclusivity_pairs(len(G.nodes))
        for (node1, node2) in mutual_exclusivity_pairs:
            model.addCons(
                zone_activation_vars[f"z{node1}"] + zone_activation_vars[f"z{node2}"] <= 1,
                name=f"MutualExclusivityConstraints_{node1}_{node2}"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
    def generate_mutual_exclusivity_pairs(self, num_nodes):
        pairs = []
        for _ in range(self.n_exclusive_pairs):
            node1 = random.randint(0, num_nodes - 1)
            node2 = random.randint(0, num_nodes - 1)
            while node1 == node2:
                node2 = random.randint(0, num_nodes - 1)
            pairs.append((node1, node2))
        return pairs

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_zones': 225,
        'max_zones': 1050,
        'er_prob': 0.31,
        'beta': 0.38,
        'n_exclusive_pairs': 200,
    }

    optimizer = NewCityPlanner(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")