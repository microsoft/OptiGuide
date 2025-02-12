import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DisasterResponseMILP:
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
            G.nodes[node]['demand'] = np.random.randint(1, 50)  # Demand for emergency aid

        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.cost_param)
            G[u][v]['terrain_factor'] = np.random.uniform(1, 3)  # Impact factor due to terrain

    def generate_population_displacement(self, G):
        population_displacement = {node: np.random.randint(50, 500) for node in G.nodes}
        return population_displacement

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        population_displacement = self.generate_population_displacement(G)
        warehouse_capacity = {node: np.random.randint(50, 200) for node in G.nodes}
        res = {'G': G, 'population_displacement': population_displacement, 'warehouse_capacity': warehouse_capacity}

        for u, v in G.edges:
            break_points = np.linspace(0, G[u][v]['cost'], self.num_pieces+1)
            slopes = np.diff(break_points)
            res[f'break_points_{u}_{v}'] = break_points
            res[f'slopes_{u}_{v}'] = slopes

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        
        model = Model("DisasterResponseMILP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piece_vars = {f"t{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"t{u}_{v}_{k}") for u, v in G.edges for k in range(self.num_pieces)}
        
        volunteer_vars = {f"z{node}": model.addVar(vtype="B", name=f"z{node}") for node in G.nodes}  # Volunteer utilization

        # Enhanced objective function
        objective_expr = quicksum(
            (G.nodes[node]['revenue'] + instance['population_displacement'][node]) * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            instance[f'slopes_{u}_{v}'][k] * piece_vars[f"t{u}_{v}_{k}"]
            for u, v in G.edges
            for k in range(self.num_pieces)
        ) + quicksum(
            G.nodes[node]['demand'] * volunteer_vars[f"z{node}"]
            for node in G.nodes
        )

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

        # Warehouse capacity constraints
        for node in G.nodes:
            model.addCons(node_vars[f"x{node}"] <= instance['warehouse_capacity'][node], name=f"WarehouseCapacity_{node}")

        # Terrain and geographical feature constraints
        for u, v in G.edges:
            model.addCons(edge_vars[f'y{u}_{v}'] <= G[u][v]['terrain_factor'], name=f"Terrain_{u}_{v}")

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
        'num_pieces': 10,
    }

    dr_milp = DisasterResponseMILP(parameters, seed=seed)
    instance = dr_milp.generate_instance()
    solve_status, solve_time = dr_milp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")