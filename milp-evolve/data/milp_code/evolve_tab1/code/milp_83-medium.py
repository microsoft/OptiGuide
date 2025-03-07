import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FCMCNF:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_commodities(self, G):
        commodities = []
        for k in range(self.n_commodities):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def generate_coverage_data(self):
        # Generate coverage data similar to set cover problem
        coverage_costs = np.random.randint(1, self.coverage_max_cost + 1, size=self.n_nodes)
        coverage_pairs = [(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes) if i != j]
        chosen_pairs = np.random.choice(len(coverage_pairs), size=self.n_coverage_pairs, replace=False)
        coverage_set = [coverage_pairs[i] for i in chosen_pairs]
        return coverage_costs, coverage_set

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        coverage_costs, coverage_set = self.generate_coverage_data()
        
        node_op_cost = np.random.uniform(self.op_cost_min, self.op_cost_max, self.n_nodes)
        warehouse_capacities = np.random.uniform(self.warehouse_cap_min, self.warehouse_cap_max, self.n_nodes)

        # Generate additional knapsack data
        item_weights = np.random.randint(self.min_item_weight, self.max_item_weight, self.total_items)
        item_profits = np.random.randint(self.min_item_profit, self.max_item_profit, self.total_items)

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings, 
            'coverage_costs': coverage_costs, 
            'coverage_set': coverage_set,
            'node_op_cost': node_op_cost,
            'warehouse_capacities': warehouse_capacities,
            'item_weights': item_weights,
            'item_profits': item_profits
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        coverage_costs = instance['coverage_costs']
        coverage_set = instance['coverage_set']
        node_op_cost = instance['node_op_cost']
        warehouse_capacities = instance['warehouse_capacities']
        item_weights = instance['item_weights']
        item_profits = instance['item_profits']
        
        model = Model("FCMCNF")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        z_vars = {f"z_{i+1}_{j+1}": model.addVar(vtype="B", name=f"z_{i+1}_{j+1}") for (i, j) in coverage_set}
        
        # New variable for node operational cost
        op_vars = {f"op_{i+1}": model.addVar(vtype="C", name=f"op_{i+1}") for i in range(self.n_nodes)}
        
        # New variables for knapsack
        k_vars = {f"k_{i+1}_{m+1}": model.addVar(vtype="B", name=f"k_{i+1}_{m+1}") for i in range(self.n_nodes) for m in range(self.total_items)}
        
        # Objective
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            coverage_costs[i] * z_vars[f"z_{i+1}_{j+1}"]
            for (i, j) in coverage_set
        )
        objective_expr += quicksum(
            node_op_cost[i] * op_vars[f"op_{i+1}"]
            for i in range(self.n_nodes)
        )
        objective_expr += quicksum(
            item_profits[m] * k_vars[f"k_{i+1}_{m+1}"]
            for i in range(self.n_nodes) for m in range(self.total_items)
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        for i in range(self.n_nodes):
            coverage_expr = quicksum(z_vars[f"z_{j+1}_{i+1}"] for j in range(self.n_nodes) if (j, i) in coverage_set)
            model.addCons(coverage_expr >= 1, f"coverage_{i+1}")

        # New constraint: operational cost should not exceed node's operational capacity
        for i in range(self.n_nodes):
            node_flow_expr = quicksum(commodities[k][2] * (quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])) for k in range(self.n_commodities))
            model.addCons(node_flow_expr <= warehouse_capacities[i], f"warehouse_capacity_{i+1}")

        # New Knapsack Constraints
        for i in range(self.n_nodes):
            knapsack_expr = quicksum(item_weights[m] * k_vars[f"k_{i+1}_{m+1}"] for m in range(self.total_items))
            model.addCons(knapsack_expr <= warehouse_capacities[i], f"knapsack_capacity_{i+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 20,
        'max_n_nodes': 30,
        'min_n_commodities': 30,
        'max_n_commodities': 45, 
        'c_range': (11, 50),
        'd_range': (10, 100),
        'ratio': 100,
        'k_max': 10,
        'er_prob': 0.3,
        'n_coverage_pairs': 50,
        'coverage_max_cost': 20,
        # New parameters for node operational costs and warehouse capacities
        'op_cost_min': 100,
        'op_cost_max': 500,
        'warehouse_cap_min': 200,
        'warehouse_cap_max': 1000,
        # New parameters for knapsack problem
        'min_item_weight': 1,
        'max_item_weight': 100,
        'min_item_profit': 10,
        'max_item_profit': 200,
        'total_items': 300
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")