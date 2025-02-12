import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ElectricalGridOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.connection_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            transmission_cost = np.random.uniform(*self.transmission_cost_range)
            maintenance_cost = np.random.uniform(self.transmission_cost_range[0] * self.maintenance_ratio, self.transmission_cost_range[1] * self.maintenance_ratio)
            capacitance = np.random.uniform(1, self.max_capacity + 1) * np.random.uniform(*self.demand_range)
            adj_mat[i, j] = (transmission_cost, maintenance_cost, capacitance)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_households(self, G):
        households = []
        for h in range(self.n_households):
            while True:
                source_substation = np.random.randint(0, self.n_nodes)
                target_household = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, source_substation, target_household) and source_substation != target_household:
                    break
            demand = int(np.random.uniform(*self.demand_range))
            households.append((source_substation, target_household, demand))
        return households

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_households = np.random.randint(self.min_n_households, self.max_n_households + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        households = self.generate_households(G)

        res = {
            'households': households, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        households = instance['households']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        
        model = Model("ElectricalGridOptimization")
        power_flow = {f"power_{i+1}_{j+1}_{h+1}": model.addVar(vtype="C", name=f"power_{i+1}_{j+1}_{h+1}") for (i, j) in edge_list for h in range(self.n_households)}
        active_lines = {f"active_{i+1}_{j+1}": model.addVar(vtype="B", name=f"active_{i+1}_{j+1}") for (i, j) in edge_list}

        objective_expr = quicksum(
            households[h][2] * adj_mat[i, j][0] * power_flow[f"power_{i+1}_{j+1}_{h+1}"]
            for (i, j) in edge_list for h in range(self.n_households)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * active_lines[f"active_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )

        for i in range(self.n_nodes):
            for h in range(self.n_households):
                delta = 1 if households[h][0] == i else -1 if households[h][1] == i else 0
                flow_expr = quicksum(power_flow[f"power_{i+1}_{j+1}_{h+1}"] for j in outcommings[i]) - quicksum(power_flow[f"power_{j+1}_{i+1}_{h+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta, f"node_balance_{i+1}_{h+1}")

        for (i, j) in edge_list:
            line_expr = quicksum(households[h][2] * power_flow[f"power_{i+1}_{j+1}_{h+1}"] for h in range(self.n_households))
            model.addCons(line_expr <= adj_mat[i, j][2] * active_lines[f"active_{i+1}_{j+1}"], f"capacity_{i+1}_{j+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 10,
        'max_n_nodes': 15,
        'min_n_households': 22,
        'max_n_households': 90,
        'transmission_cost_range': (33, 150),
        'demand_range': (90, 900),
        'maintenance_ratio': 1500,
        'max_capacity': 30,
        'connection_prob': 0.52,
    }

    electrical_grid_optimization = ElectricalGridOptimization(parameters, seed=seed)
    instance = electrical_grid_optimization.generate_instance()
    solve_status, solve_time = electrical_grid_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")