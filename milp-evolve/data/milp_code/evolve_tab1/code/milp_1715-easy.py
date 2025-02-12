import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GISP:
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
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        if self.set_type == 'SET1':
            for node in G.nodes:
                G.nodes[node]['revenue'] = np.random.randint(1, 100)
            for u, v in G.edges:
                G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.set_param)
        elif self.set_type == 'SET2':
            for node in G.nodes:
                G.nodes[node]['revenue'] = float(self.set_param)
            for u, v in G.edges:
                G[u][v]['cost'] = 1.0

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
        
        # Simulating energy data
        energy_cost = {node: np.random.uniform(5, 20) for node in G.nodes}
        total_energy_limit = np.sum([energy_cost[node] for node in G.nodes]) * 0.75  # Limiting to 75% of max energy
        
        res = {
            'G': G,
            'E2': E2,
            'energy_cost': energy_cost,
            'total_energy_limit': total_energy_limit
        }
        # New instance data (Big M method)
        upper_energy_limit = total_energy_limit * 1.1  # Allow 10% over limit
        energy_penalty = 1000  # Penalty for exceeding energy limit per unit
        
        res.update({
            'upper_energy_limit': upper_energy_limit,
            'energy_penalty': energy_penalty
        })
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']
        energy_cost = instance['energy_cost']
        total_energy_limit = instance['total_energy_limit']
        upper_energy_limit = instance['upper_energy_limit']
        energy_penalty = instance['energy_penalty']
        
        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        exceeding_energy = model.addVar(vtype="C", name="exceeding_energy")

        # Objective: Maximize revenue minus cost of removing removable edges, including penalties for energy violations
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        objective_expr -= energy_penalty * exceeding_energy

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

        # Constraints: Energy
        energy_consumed = quicksum(energy_cost[node] * node_vars[f"x{node}"] for node in G.nodes)
        model.addCons(
            energy_consumed <= total_energy_limit,
            name="Total_Energy_Limit"
        )
        
        # Big M constraint for energy consumption exceeding penalty
        M = upper_energy_limit
        model.addCons(
            energy_consumed - total_energy_limit <= exceeding_energy + M,
            name="Energy_Exceeding_Constraint"
        )
        
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 100,
        'max_n': 130,
        'er_prob': 0.78,
        'set_type': 'SET1',
        'set_param': 2000.0,
        'alpha': 0.77,
    }
    # New parameter code
    parameters.update({
        'upper_energy_limit': parameters['max_n'] * 20 * 1.1,  # Example value for new parameter
        'energy_penalty': 1000,  # Example value for new parameter
    })

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")