import random
import time
import numpy as np
import networkx as nx
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
        return Graph(number_of_nodes, edges, degrees, neighbors)

    def efficient_greedy_clique_partition(self):
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()
        while leftover_nodes:
            clique_center = leftover_nodes[0]
            leftover_nodes = leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]
        return cliques

class SupplyChainNetworkDesign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)
        return G, adj_mat, edge_list, incommings, outcommings

    def generate_independent_set_graph(self):
        graph = Graph.erdos_renyi(self.n_nodes, self.edge_prob_ind_set)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
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
        
        return graph, inequalities

    def generate_commodities(self, G):
        commodities = []
        for k in range(self.n_commodities):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            supply_k = int(np.random.uniform(*self.s_range))  # Changed from demand to supply
            commodities.append((o_k, d_k, supply_k))
        return commodities

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        
        graph, inequalities = self.generate_independent_set_graph()

        instance = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'graph': graph,
            'inequalities': inequalities
        }
        return instance

    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        graph = instance['graph']
        inequalities = instance['inequalities']
        
        model = Model("SupplyChainNetworkDesign")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        z_vars = {f"z_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"z_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}

        node_vars = {i: model.addVar(vtype="B", name=f"node_{i+1}") for i in range(self.n_nodes)}

        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            10.0 * node_vars[i] for i in range(self.n_nodes)
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

            for k in range(self.n_commodities):
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] >= x_vars[f"x_{i+1}_{j+1}_{k+1}"] - (1 - y_vars[f"y_{i+1}_{j+1}"]) * adj_mat[i, j][2], f"comp1_{i+1}_{j+1}_{k+1}")
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] <= x_vars[f"x_{i+1}_{j+1}_{k+1}"], f"comp2_{i+1}_{j+1}_{k+1}")
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] <= y_vars[f"y_{i+1}_{j+1}"] * adj_mat[i, j][2], f"comp3_{i+1}_{j+1}_{k+1}")

        for count, group in enumerate(inequalities):
            model.addCons(quicksum(node_vars[node] for node in group) <= 1, name=f"clique_{count}")

        for i in range(self.n_nodes):
            model.addCons(node_vars[i] <= quicksum(y_vars[f"y_{i+1}_{j+1}"] for j in outcommings[i]), name=f"node_edge_{i+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 20,
        'max_n_nodes': 30,
        'min_n_commodities': 30,
        'max_n_commodities': 45, 
        'c_range': (11, 50),
        'd_range': (10, 100),
        'ratio': 100,
        'k_max': 10,
        'er_prob': 0.3,
        'edge_prob_ind_set': 0.25,
        's_range': (10, 100)
    }

    fcmcnf = SupplyChainNetworkDesign(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")