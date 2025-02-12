import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum


class ManufacturingMILP:
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

    def generate_machine_capacities(self, G):
        for node in G.nodes:
            G.nodes[node]['capacity'] = np.random.randint(50, 200)

    def generate_maintenance_schedule(self, G):
        for node in G.nodes:
            G.nodes[node]['maintenance'] = np.random.choice([0, 1], p=[0.8, 0.2])

    def generate_demand(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.normal(self.mean_demand, self.std_dev_demand)

    def generate_carbon_footprint(self, G):
        for u, v in G.edges:
            G[u][v]['carbon_footprint'] = np.random.uniform(0.1, 2.0)

    def generate_renewable_energy_limits(self):
        self.renewable_energy_limit = np.random.uniform(100, 500)

    def generate_market_prices(self):
        self.variable_costs = np.random.uniform(1, 10, size=self.max_n)

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_machine_capacities(G)
        self.generate_maintenance_schedule(G)
        self.generate_demand(G)
        self.generate_carbon_footprint(G)
        self.generate_renewable_energy_limits()
        self.generate_market_prices()

        return {'G': G, 'renewable_energy_limit': self.renewable_energy_limit, 'variable_costs': self.variable_costs}
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        renewable_energy_limit = instance['renewable_energy_limit']
        variable_costs = instance['variable_costs']
        
        model = Model("Manufacturing_Optimization")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        
        objective_expr = quicksum(
            (G.nodes[node]['demand'] * variable_costs[node]) * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            G[u][v]['carbon_footprint'] * edge_vars[f"y{u}_{v}"]
            for u, v in G.edges
        )

        for u, v in G.edges:
            # Conflict-free node selection
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                name=f"C_{u}_{v}"
            )

            # If an edge is selected, at least one of its nodes must be selected
            model.addCons(
                edge_vars[f"y{u}_{v}"] <= node_vars[f"x{u}"] + node_vars[f"x{v}"],
                name=f"Logical_Cond1_{u}_{v}"
            )

            # Carbon footprint minimization
            model.addCons(
                edge_vars[f"y{u}_{v}"] * G[u][v]['carbon_footprint'] <= renewable_energy_limit,
                name=f"Carbon_Footprint_{u}_{v}"
            )

        # Maintenance scheduling
        for node in G.nodes:
            if G.nodes[node]['maintenance'] == 1:
                model.addCons(
                    node_vars[f"x{node}"] == 0,
                    name=f"Maintenance_{node}"
                )

        # Renewable energy constraint
        model.addCons(
            quicksum(node_vars[f"x{node}"] for node in G.nodes) <= renewable_energy_limit,
            name="Renewable_Energy_Limit"
        )

        # Production capacities not exceeded
        for node in G.nodes:
            model.addCons(
                node_vars[f"x{node}"] <= G.nodes[node]['capacity'],
                name=f"Capacity_{node}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 350,
        'max_n': 500,
        'er_prob': 0.76,
        'mean_demand': 120,
        'std_dev_demand': 500,
    }

    parameters.update({'renewable_energy_limit': 300}) 

    manufacturing_milp = ManufacturingMILP(parameters, seed=seed)
    instance = manufacturing_milp.generate_instance()
    solve_status, solve_time = manufacturing_milp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")