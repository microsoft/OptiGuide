import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedGISP:
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
        if self.graph_type == 'ER':
            G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        elif self.graph_type == 'BA':
            G = nx.barabasi_albert_graph(n=n_nodes, m=self.barabasi_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
            G.nodes[node]['penalty'] = np.random.randint(1, 50)

        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.cost_param)

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        res = {'G': G}

        for u, v in G.edges:
            break_points = np.linspace(0, G[u][v]['cost'], self.num_pieces+1)
            slopes = np.diff(break_points)
            res[f'break_points_{u}_{v}'] = break_points
            res[f'slopes_{u}_{v}'] = slopes

        for node in G.nodes:
            res[f'fixed_cost_{node}'] = np.random.randint(100, 200)
            res[f'variable_cost_{node}'] = np.random.uniform(1, 5)

        for u, v in G.edges:
            res[f'max_trans_cost_{u}_{v}'] = np.random.uniform(10, 20)
            res[f'min_trans_cost_{u}_{v}'] = np.random.uniform(5, 10)

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']

        model = Model("SimplifiedGISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piece_vars = {f"t{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"t{u}_{v}_{k}") for u, v in G.edges for k in range(self.num_pieces)}

        # Modified objective function
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            instance[f'slopes_{u}_{v}'][k] * piece_vars[f"t{u}_{v}_{k}"]
            for u, v in G.edges
            for k in range(self.num_pieces)
        )

        # Applying Piecewise Linear Function Constraints
        M = 10000  # Big M constant, should be large enough
        for u, v in G.edges:
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] * M <= 1,
                name=f"C_{u}_{v}"
            )

        for u, v in G.edges:
            model.addCons(quicksum(piece_vars[f"t{u}_{v}_{k}"] for k in range(self.num_pieces)) == edge_vars[f'y{u}_{v}'])

            for k in range(self.num_pieces):
                model.addCons(piece_vars[f"t{u}_{v}_{k}"] <= instance[f'break_points_{u}_{v}'][k+1] - instance[f'break_points_{u}_{v}'][k] + (1 - edge_vars[f'y{u}_{v}']) * M)

        for node in G.nodes:
            model.addCons(
                node_vars[f"x{node}"] * instance[f'fixed_cost_{node}'] + node_vars[f"x{node}"] * instance[f'variable_cost_{node}'] <= self.transport_budget
            )

        # Define new semi-continuous variables
        semicon_vars = {f"s{u}_{v}": model.addVar(vtype="C", name=f"s{u}_{v}") for u, v in G.edges}

        # Add constraints for semi-continuous variables
        for u, v in G.edges:
            max_trans_cost = instance[f'max_trans_cost_{u}_{v}']
            min_trans_cost = instance[f'min_trans_cost_{u}_{v}']
            model.addCons(semicon_vars[f"s{u}_{v}"] >= edge_vars[f"y{u}_{v}"] * min_trans_cost)
            model.addCons(semicon_vars[f"s{u}_{v}"] <= edge_vars[f"y{u}_{v}"] * max_trans_cost)
            objective_expr -= semicon_vars[f"s{u}_{v}"]

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 81,
        'max_n': 1024,
        'er_prob': 0.1,
        'graph_type': 'BA',
        'barabasi_m': 25,
        'cost_param': 3000.0,
        'num_pieces': 20,
        'transport_budget': 50000,
    }

    params = {
        'max_trans_cost': 20.00,
        'min_trans_cost': 5.00,
    }
    parameters.update(params)

    gisp = SimplifiedGISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")