import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ModifiedGISP:
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
            G[u][v]['penalty'] = np.random.uniform(0.5, 2.0)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        res = {'G': G, 'E2': E2}
        
        for u, v in G.edges:
            break_points = np.linspace(0, G[u][v]['cost'], self.num_pieces+1)
            slopes = np.diff(break_points)
            res[f'break_points_{u}_{v}'] = break_points
            res[f'slopes_{u}_{v}'] = slopes
        
        for node in G.nodes:
            res[f'fixed_cost_{node}'] = np.random.randint(100, 200)
            res[f'variable_cost_{node}'] = np.random.uniform(1, 5)
            res[f'emission_{node}'] = np.random.uniform(0.1, 0.5)

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        
        model = Model("ModifiedGISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piece_vars = {f"t{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"t{u}_{v}_{k}") for u, v in G.edges for k in range(self.num_pieces)}
        extra_vars = {f"z{node}": model.addVar(vtype="I", lb=0, ub=10, name=f"z{node}") for node in G.nodes}
        
        penalty_vars = {f"penalty_{node}": model.addVar(vtype="C", lb=0, name=f"penalty_{node}") for node in G.nodes}

        # New variable to impose a limit on total emissions
        emission_limit = model.addVar(vtype="C", lb=0, ub=1, name="emission_limit")

        # New variables for vehicle utilization
        utilization_vars = {f"utilization_{node}": model.addVar(vtype="I", lb=0, name=f"utilization_{node}") for node in G.nodes}

        # New emission constraint vars
        emission_vars = {f"emission_{node}": model.addVar(vtype="C", lb=0, name=f"emission_{node}") for node in G.nodes}

        # Modify objective function as per the new elements
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"] - G.nodes[node]['penalty'] * extra_vars[f"z{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            instance[f'slopes_{u}_{v}'][k] * piece_vars[f"t{u}_{v}_{k}"]
            for u, v in E2
            for k in range(self.num_pieces)
        )
        
        objective_expr += quicksum(
            G[u][v]['penalty'] * edge_vars[f'y{u}_{v}']
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            penalty_vars[f"penalty_{node}"] for node in G.nodes
        )

        # Applying Piecewise Linear Function Constraints
        M = 10000  # Big M constant, should be large enough
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] * M <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] * M <= M,
                    name=f"C_{u}_{v}"
                )
                
        for u, v in E2:
            model.addCons(quicksum(piece_vars[f"t{u}_{v}_{k}"] for k in range(self.num_pieces)) == edge_vars[f'y{u}_{v}'])
            
            for k in range(self.num_pieces):
                model.addCons(piece_vars[f"t{u}_{v}_{k}"] <= instance[f'break_points_{u}_{v}'][k+1] - instance[f'break_points_{u}_{v}'][k] + (1 - edge_vars[f'y{u}_{v}']) * M)

        for node in G.nodes:
            model.addCons(extra_vars[f"z{node}"] <= 10 * node_vars[f"x{node}"], name=f"Cons_z_{node}_bigM")
            model.addCons(extra_vars[f"z{node}"] >= 0, name=f"Cons_z_{node}")

            # Utilization constraint
            model.addCons(utilization_vars[f"utilization_{node}"] >= 2 * node_vars[f"x{node}"])
            model.addCons(utilization_vars[f"utilization_{node}"] <= 5 * node_vars[f"x{node}"])

            # Penalty for exceeding distance
            model.addCons(penalty_vars[f"penalty_{node}"] >= G.nodes[node]['penalty'] * (1 - node_vars[f"x{node}"]))

            # Emission constraints
            model.addCons(emission_vars[f"emission_{node}"] <= instance[f'emission_{node}'] * node_vars[f"x{node}"])

        # Budget constraint: Fixed + Variable costs
        total_budget = quicksum(instance[f'fixed_cost_{node}'] * node_vars[f"x{node}"] +
                                instance[f'variable_cost_{node}'] * node_vars[f"x{node}"]
                                for node in G.nodes)
        model.addCons(total_budget <= self.transport_budget)

        # Emission reduction constraint
        total_emissions = quicksum(emission_vars[f"emission_{node}"] for node in G.nodes)
        model.addCons(total_emissions <= emission_limit)

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 12,
        'max_n': 2700,
        'er_prob': 0.52,
        'graph_type': 'BA',
        'barabasi_m': 5,
        'cost_param': 3000.0,
        'alpha': 0.17,
        'num_pieces': 112,
        'transport_budget': 50000,
    }

    gisp = ModifiedGISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")