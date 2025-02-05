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

    def find_maximal_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        cliques = self.find_maximal_cliques(G)
        
        # Extend to include new data
        inventory_holding_costs = {node: np.random.randint(1, 10) for node in G.nodes}
        renewable_energy_costs = {(u, v): np.random.randint(1, 10) for u, v in G.edges}
        carbon_emissions = {(u, v): np.random.randint(1, 10) for u, v in G.edges}
        
        return {'G': G, 'E2': E2, 'cliques': cliques, 
                'inventory_holding_costs': inventory_holding_costs, 
                'renewable_energy_costs': renewable_energy_costs,
                'carbon_emissions': carbon_emissions}
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, cliques = instance['G'], instance['E2'], instance['cliques']
        inventory_holding_costs = instance['inventory_holding_costs']
        renewable_energy_costs = instance['renewable_energy_costs']
        carbon_emissions = instance['carbon_emissions']
        
        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piecewise_vars = {f"z{u}_{v}": model.addVar(vtype="B", name=f"z{u}_{v}") for u, v in G.edges}

        # New variables for inventory, demand, renewable energy, and carbon emissions
        inventory_level = {f"inv_{node}": model.addVar(vtype="C", name=f"inv_{node}") for node in G.nodes}
        demand_level = {f"dem_{node}": model.addVar(vtype="C", name=f"dem_{node}") for node in G.nodes}
        renewable_energy_vars = {f"re_{u}_{v}": model.addVar(vtype="B", name=f"re_{u}_{v}") for u, v in G.edges}

        # Objective function: Maximize revenue, minimize costs
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        objective_expr -= quicksum(
            inventory_holding_costs[node] * inventory_level[f"inv_{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            renewable_energy_costs[(u, v)] * renewable_energy_vars[f"re_{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            carbon_emissions[(u, v)] * edge_vars[f"y{u}_{v}"]
            for u, v in G.edges
        )

        # Existing constraints
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

        # Adding clique constraints
        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(node_vars[f"x{node}"] for node in clique) <= 1,
                name=f"Clique_{i}"
            )

        # Piecewise Linear Functions Constraints
        for u, v in G.edges:
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1 + piecewise_vars[f"z{u}_{v}"],
                name=f"PL1_{u}_{v}"
            )
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] >= 2 * piecewise_vars[f"z{u}_{v}"],
                name=f"PL2_{u}_{v}"
            )

        # New renewable energy constraints
        for u, v in G.edges:
            model.addCons(
                edge_vars[f"y{u}_{v}"] >= renewable_energy_vars[f"re_{u}_{v}"],
                name=f"RE_{u}_{v}"
            )
        
        # New inventory constraints
        for node in G.nodes:
            model.addCons(
                inventory_level[f"inv_{node}"] >= 0,
                name=f"Inv_{node}_nonneg"
            )
            model.addCons(
                inventory_level[f"inv_{node}"] - demand_level[f"dem_{node}"] >= 0,
                name=f"Stock_{node}_out"
            )
        
        # Objective and solve
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 50,
        'max_n': 975,
        'er_prob': 0.23,
        'set_type': 'SET1',
        'set_param': 225.0,
        'alpha': 0.73,
        'renewable_percentage': 0.5,
        'carbon_limit': 2000,
    }
    
    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")