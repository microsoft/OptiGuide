import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexGISP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale)
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.lognormal(mean=self.lognormal_mean, sigma=self.lognormal_sigma)
            G[u][v]['compatibility_cost'] = np.random.randint(1, 20)
            G[u][v]['capacity'] = np.random.randint(1, 10)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_incompatible_pairs(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.beta:
                E_invalid.add(edge)
        return E_invalid

    def generate_bids_and_exclusivity(self, G):
        cliques = list(nx.find_cliques(G.to_undirected()))
        bids = [(clique, np.random.uniform(50, 200)) for clique in cliques]
        exclusivity_pairs = set()
        while len(exclusivity_pairs) < self.n_exclusive_pairs:
            bid1 = np.random.randint(0, len(bids))
            bid2 = np.random.randint(0, len(bids))
            if bid1 != bid2:
                exclusivity_pairs.add((bid1, bid2))
        return bids, exclusivity_pairs

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        E_invalid = self.generate_incompatible_pairs(G)
        bids, exclusivity_pairs = self.generate_bids_and_exclusivity(G)
        res = {
            'G': G, 
            'E2': E2, 
            'E_invalid': E_invalid, 
            'bids': bids, 
            'exclusivity_pairs': exclusivity_pairs
        }
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, E_invalid = instance['G'], instance['E2'], instance['E_invalid']
        bids, exclusivity_pairs = instance['bids'], instance['exclusivity_pairs']
        
        model = Model("Complex_GISP")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        load_vars = {f"load_{node}": model.addVar(vtype="I", name=f"load_{node}") for node in G.nodes}
        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(bids))}

        # Objective: Maximize revenues, minimize costs including compatibility costs, and balance loads
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )
        objective_expr -= quicksum(
            load_vars[f"load_{node}"]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['compatibility_cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E_invalid
        )
        objective_expr += quicksum(price * bid_vars[i] for i, (bundle, price) in enumerate(bids))

        # Constraints from Enhanced_GISP
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )

        # Set Partitioning Constraint
        for node in G.nodes:
            model.addCons(
                quicksum(edge_vars[f"y{u}_{v}"] for u, v in E2 if u == node or v == node) == node_vars[f"x{node}"],
                name=f"Partition_{node}"
            )

        # Load Balancing Constraints
        for node in G.nodes:
            model.addCons(
                load_vars[f"load_{node}"] == quicksum(edge_vars[f"y{u}_{v}"] for u, v in E2 if u == node or v == node),
                name=f"Load_{node}"
            )

        # Compatibility Constraints
        for u, v in G.edges:
            if (u, v) in E_invalid:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"Incompatible_{u}_{v}"
                )

        # Mutual exclusivity constraints for bids
        for (bid1, bid2) in exclusivity_pairs:
            model.addCons(bid_vars[bid1] + bid_vars[bid2] <= 1, f"Exclusive_{bid1}_{bid2}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 1050,
        'max_n': 1560,
        'ba_m': 3,
        'set_type': 'SET1',
        'alpha': 0.31,
        'beta': 0.8,
        'gamma_shape': 72.0,
        'gamma_scale': 60.0,
        'lognormal_mean': 0.0,
        'lognormal_sigma': 4.5,
        'n_exclusive_pairs': 14,
    }

    complex_gisp = ComplexGISP(parameters, seed=seed)
    instance = complex_gisp.generate_instance()
    solve_status, solve_time = complex_gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")