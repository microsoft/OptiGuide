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
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.gamma(shape=self.gamma_shape, scale=self.gamma_scale)
            G.nodes[node]['capacity'] = np.random.randint(1, self.max_capacity)
        for u, v in G.edges:
            G[u][v]['cost'] = np.random.normal(loc=self.norm_mean, scale=self.norm_sd)

    def generate_tasks(self):
        num_tasks = np.random.randint(self.min_tasks, self.max_tasks)
        tasks = [np.random.randint(1, self.max_task_capacity) for _ in range(num_tasks)]
        return tasks

    def generate_special_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        tasks = self.generate_tasks()
        E2 = self.generate_special_edges(G)

        facility_costs = np.random.randint(1, self.max_cost, size=len(G))     
        node_pairs_cost = np.asarray(nx.floyd_warshall_numpy(G))

        res = {'G': G, 'tasks': tasks, 'E2': E2, 'facility_costs': facility_costs, 'node_pairs_cost': node_pairs_cost}
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, tasks, E2, facility_costs, node_pairs_cost = instance['G'], instance['tasks'], instance['E2'], instance['facility_costs'], instance['node_pairs_cost']
        
        model = Model("EnhancedGISP")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        task_vars = {f"t{task}": model.addVar(vtype="B", name=f"t{task}") for task in range(len(tasks))}
        special_edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in E2}
        facility_vars = {f"f{node}": model.addVar(vtype="B", name=f"f{node}", obj=facility_costs[node]) for node in G.nodes}

        # Complex objective with weighted sum of node revenues, task covers and edge costs
        objective_expr = quicksum(
            (G.nodes[node]['revenue'] * node_vars[f"x{node}"])
            for node in G.nodes
        ) + quicksum(
            task_vars[f"t{task}"]
            for task in range(len(tasks))
        ) - quicksum(
            (G[u][v]['cost'] * special_edge_vars[f"y{u}_{v}"])
            for u, v in E2
        ) - quicksum(
            (facility_costs[node] * facility_vars[f"f{node}"])
            for node in G.nodes
        )
        model.setObjective(objective_expr, "maximize")

        # Replacing capacity constraints with set covering constraints
        for task in range(len(tasks)):
            model.addCons(
                quicksum(G.nodes[node]['capacity'] * node_vars[f"x{node}"] for node in G.nodes) >= tasks[task],
                name=f"Cover_Task_{task}"
            )

        # Ensure maximum budget constraint for facility placement
        model.addCons(
            quicksum(facility_vars[f"f{node}"] * facility_costs[node] for node in G.nodes) <= self.max_budget,
            name="MaxBudget"
        )

        # Ensure coverage constraint for every node
        for i in G.nodes:
            model.addCons(
                quicksum(facility_vars[f"f{j}"] for j in G.nodes if node_pairs_cost[i][j] <= self.cover_threshold) >= 1,
                name=f"Coverage_{i}"
            )

        # Enhanced constraints considering node degrees and special edge interactions
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - special_edge_vars[f"y{u}_{v}"] + quicksum(node_vars[f"x{u}"] for u in G.neighbors(u)) <= 2,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 135,
        'max_n': 1040,
        'ba_m': 63,
        'gamma_shape': 67.5,
        'gamma_scale': 1.0,
        'norm_mean': 0.0,
        'norm_sd': 37.5,
        'max_capacity': 600,
        'alpha': 0.17,
        'min_tasks': 50,
        'max_tasks': 1000,
        'max_task_capacity': 750,
        'max_cost': 200,
        'max_budget': 750,
        'cover_threshold': 14,
    }

    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")