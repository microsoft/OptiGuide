import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class AdvancedGISP:
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
            res[f'weight_{node}'] = np.random.uniform(1, 10) # new weight for knapsack

        for u, v in G.edges:
            res[f'weight_{u}_{v}'] = np.random.uniform(0.5, 5) # new weight for knapsack

        res['set_pack_penalties'] = np.random.uniform(30, 100, len(G.nodes)) # new set pack penalties for complexity

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        
        model = Model("AdvancedGISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piece_vars = {f"t{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"t{u}_{v}_{k}") for u, v in G.edges for k in range(self.num_pieces)}
        z_knapsack = {f"z{node}": model.addVar(vtype="B", name=f"z{node}") for node in G.nodes} # new knapsack variables
        set_pack_vars = {f"p{node}": model.addVar(vtype="B", name=f"p{node}") for node in G.nodes} # new variables for set packing
        
        # Modified objective function
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            instance[f'slopes_{u}_{v}'][k] * piece_vars[f"t{u}_{v}_{k}"]
            for u, v in G.edges
            for k in range(self.num_pieces)
        ) - quicksum(
            instance['set_pack_penalties'][i] * set_pack_vars[f"p{i}"]
            for i, node in enumerate(G.nodes)
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

        # New Knapsack Constraints
        model.addCons(
            quicksum(instance[f'weight_{node}'] * z_knapsack[f"z{node}"] for node in G.nodes) <= self.knapsack_capacity,
            name="KnapsackNode"
        )

        model.addCons(
            quicksum(instance[f'weight_{u}_{v}'] * edge_vars[f"y{u}_{v}"] for u, v in G.edges) <= self.knapsack_capacity,
            name="KnapsackEdge"
        )

        # New Set Packing Constraints
        for node in G.nodes:
            for neighbor in G.neighbors(node):
                model.addCons(
                    node_vars[f"x{node}"] + node_vars[f"x{neighbor}"] <= 1,
                    name=f"SetPacking_{node}_{neighbor}"
                )
        
        # New Constraints for Set Packing Penalties
        for node in G.nodes:
            model.addCons(
                set_pack_vars[f"p{node}"] <= node_vars[f"x{node}"],
                name=f"SetPackPen_{node}"
            )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 108,
        'max_n': 2048,
        'er_prob': 0.58,
        'graph_type': 'BA',
        'barabasi_m': 5,
        'cost_param': 3000.0,
        'num_pieces': 10,
        'transport_budget': 50000,
        'knapsack_capacity': 1000, # new parameter for knapsack capacity
    }
    gisp = AdvancedGISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")