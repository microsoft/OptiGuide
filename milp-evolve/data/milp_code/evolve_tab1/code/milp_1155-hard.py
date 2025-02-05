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
        
        weights = np.random.randint(self.min_range, self.max_range, self.knapsack_items)
        profits = np.random.randint(self.min_range, self.max_range, self.knapsack_items)
        capacities = np.zeros(self.knapsack_count, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.knapsack_count,
                                            0.6 * weights.sum() // self.knapsack_count,
                                            self.knapsack_count - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        res = {'G': G, 'E2': E2, 'weights': weights, 'profits': profits, 'capacities': capacities}
        
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, weights, profits, capacities = instance['G'], instance['E2'], instance['weights'], instance['profits'], instance['capacities']
        
        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        aux_vars = {f"z{u}_{v}": model.addVar(vtype="B", name=f"z{u}_{v}") for u, v in G.edges if (u, v) not in E2}
        knapsack_vars = {f"k{item}": model.addVar(vtype="B", name=f"k{item}") for item in range(len(profits))}
        
        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )
        objective_expr += quicksum(profits[item] * knapsack_vars[f"k{item}"] for item in range(len(profits)))

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
                model.addCons(
                    aux_vars[f"z{u}_{v}"] <= node_vars[f"x{u}"],
                    name=f"CH1_{u}_{v}"
                )
                model.addCons(
                    aux_vars[f"z{u}_{v}"] <= node_vars[f"x{v}"],
                    name=f"CH2_{u}_{v}"
                )
                model.addCons(
                    aux_vars[f"z{u}_{v}"] >= node_vars[f"x{u}"] + node_vars[f"x{v}"] - 1,
                    name=f"CH3_{u}_{v}"
                )
                objective_expr -= aux_vars[f"z{u}_{v}"]

        # Knapsack constraints
        for c in range(len(capacities)):
            model.addCons(
                quicksum(weights[i] * knapsack_vars[f"k{i}"] for i in range(len(weights))) <= capacities[c],
                f"KnapsackCapacity_{c}"
            )

        # Ensure each item is assigned to at most one knapsack
        for item in range(len(weights)):
            model.addCons(
                knapsack_vars[f"k{item}"] <= 1
            )

        # Logical conditions
        item_A, item_B = 0, 1
        model.addCons(
            knapsack_vars[f"k{item_A}"] <= knapsack_vars[f"k{item_B}"],
            "LogicalCondition_1"
        )

        item_C, item_D = 2, 3
        for c in range(len(capacities)):
            model.addCons(
                knapsack_vars[f"k{item_C}"] == knapsack_vars[f"k{item_D}"],
                f"LogicalCondition_2_{c}"
            )

        item_E, item_F = 4, 5
        for c in range(len(capacities)):
            model.addCons(
                knapsack_vars[f"k{item_E}"] + knapsack_vars[f"k{item_F}"] <= 1,
                f"LogicalCondition_3_{c}"
            )
        
        min_items_per_knapsack = 2
        for c in range(len(capacities)):
            model.addCons(
                quicksum(knapsack_vars[f"k{i}"] for i in range(len(weights))) >= min_items_per_knapsack,
                f"LogicalCondition_4_{c}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 75,
        'max_n': 260,
        'er_prob': 0.73,
        'set_type': 'SET1',
        'set_param': 700.0,
        'alpha': 0.31,
        'knapsack_items': 400,
        'knapsack_count': 50,
        'min_range': 70,
        'max_range': 150,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")