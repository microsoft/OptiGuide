import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class REDO:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_random_network(self):
        n_communities = np.random.randint(self.min_communities, self.max_communities)
        G = nx.erdos_renyi_graph(n=n_communities, p=self.er_prob, seed=self.seed)
        return G

    def generate_energy_data(self, G):
        for node in G.nodes:
            G.nodes[node]['energy_need'] = np.random.randint(1, 100)
            G.nodes[node]['supply_variance'] = np.random.randint(5, 15)  # Variance in supply

        for u, v in G.edges:
            G[u][v]['segments'] = [((i + 1) * 10, np.random.randint(1, 10)) for i in range(self.num_segments)]
            G[u][v]['capacity'] = np.random.randint(50, 200)  # Energy transportation capacity

    def find_generation_zones(self, G):
        cliques = list(nx.find_cliques(G))
        generation_zones = [clique for clique in cliques if len(clique) > 1]
        return generation_zones

    def generate_instance(self):
        G = self.generate_random_network()
        self.generate_energy_data(G)
        zones = self.find_generation_zones(G)

        return {
            'G': G,
            'zones': zones,
        }

    def solve(self, instance):
        G, zones = instance['G'], instance['zones']

        model = Model("REDO")

        # Variables
        need_vars = {f"n{node}": model.addVar(vtype="B", name=f"n{node}") for node in G.nodes}
        capacity_vars = {(u, v): model.addVar(vtype="I", name=f"capacity_{u}_{v}") for u, v in G.edges}
        cost_vars = {(u, v): model.addVar(vtype="C", name=f"cost_{u}_{v}") for u, v in G.edges}

        segment_vars = {}
        for u, v in G.edges:
            for i in range(self.num_segments):
                segment_vars[(u, v, i)] = model.addVar(vtype="C", name=f"segment_{u}_{v}_{i}")
        
        energy_vars = {(u, v): model.addVar(vtype="B", name=f"energy_{u}_{v}") for u, v in G.edges}
        penalty_vars = {node: model.addVar(vtype="C", name=f"penalty_{node}") for node in G.nodes}

        # Objective
        objective_expr = quicksum(G.nodes[node]['energy_need'] * need_vars[f"n{node}"] for node in G.nodes)
        for u, v in G.edges:
            for i, (amount, cost) in enumerate(G[u][v]['segments']):
                objective_expr -= segment_vars[(u, v, i)] * cost
        objective_expr -= quicksum(penalty_vars[node] for node in G.nodes)

        model.setObjective(objective_expr, "maximize")

        # Constraints
        for u, v in G.edges:
            model.addCons(
                need_vars[f"n{u}"] + need_vars[f"n{v}"] <= 1,
                name=f"Need_{u}_{v}"
            )
            model.addCons(
                quicksum(segment_vars[(u, v, i)] for i in range(self.num_segments)) == energy_vars[(u, v)] * 100,
                name=f"Segment_{u}_{v}"
            )
            model.addCons(
                quicksum(segment_vars[(u, v, i)] for i in range(self.num_segments)) <= capacity_vars[(u, v)],
                name=f"Capacity_{u}_{v}"
            )

        for i, zone in enumerate(zones):
            model.addCons(
                quicksum(need_vars[f"n{community}"] for community in zone) <= 1,
                name=f"Supply_{i}"
            )

        for node in G.nodes:
            model.addCons(
                sum(energy_vars[(u, v)] for u, v in G.edges if u == node or v == node) * G.nodes[node]['supply_variance'] >= penalty_vars[node],
                name=f"Penalty_{node}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_communities': 50,
        'max_communities': 1050,
        'er_prob': 0.31,
        'num_segments': 1,
    }

    redo = REDO(parameters, seed=seed)
    instance = redo.generate_instance()
    solve_status, solve_time = redo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")