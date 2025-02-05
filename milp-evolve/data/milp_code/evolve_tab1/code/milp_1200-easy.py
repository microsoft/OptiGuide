import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DRRNetwork:
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

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        hub_costs = np.random.randint(self.min_hub_cost, self.max_hub_cost + 1, self.n_nodes)
        connection_costs = np.random.randint(self.min_connection_cost, self.max_connection_cost + 1, (self.n_nodes, self.n_nodes))
        capacities = np.random.randint(self.min_hub_capacity, self.max_hub_capacity + 1, self.n_nodes)
        
        maintenance_costs = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost + 1, (self.n_nodes, self.n_nodes))
        zones = np.random.randint(0, self.max_zones, self.n_nodes)

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'hub_costs': hub_costs,
            'connection_costs': connection_costs,
            'capacities': capacities,
            'maintenance_costs': maintenance_costs,
            'zones': zones,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        
        hub_costs = instance['hub_costs']
        connection_costs = instance['connection_costs']
        capacities = instance['capacities']
        maintenance_costs = instance['maintenance_costs']
        zones = instance['zones']

        model = Model("DRRNetwork")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        hub_vars = {f"Hub_{i+1}": model.addVar(vtype="B", name=f"Hub_{i+1}") for i in range(self.n_nodes)}
        connection_vars = {f"Connection_{i+1}_{j+1}": model.addVar(vtype="B", name=f"Connection_{i+1}_{j+1}") for i in range(self.n_nodes) for j in range(self.n_nodes)}
        
        # New Maintenance cost variables
        maintenance_vars = {f"Maintenance_{i+1}_{j+1}": model.addVar(vtype="C", name=f"Maintenance_{i+1}_{j+1}") for (i, j) in edge_list}

        # Update objective function to include maintenance costs
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            hub_costs[i] * hub_vars[f"Hub_{i+1}"]
            for i in range(self.n_nodes)
        )
        objective_expr += quicksum(
            connection_costs[i, j] * connection_vars[f"Connection_{i+1}_{j+1}"]
            for i in range(self.n_nodes) for j in range(self.n_nodes)
        )
        objective_expr += quicksum(
            maintenance_costs[i, j] * maintenance_vars[f"Maintenance_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            M = sum(commodities[k][2] for k in range(self.n_commodities))  # Big M
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")
            model.addCons(arc_expr <= M * y_vars[f"y_{i+1}_{j+1}"], f"big_M_arc_{i+1}_{j+1}")

        # Set covering constraints: Ensure that every node is covered
        for i in range(self.n_nodes):
            cover_expr_in = quicksum(y_vars[f"y_{j+1}_{i+1}"] for j in incommings[i])
            cover_expr_out = quicksum(y_vars[f"y_{i+1}_{j+1}"] for j in outcommings[i])
            model.addCons(cover_expr_in + cover_expr_out >= 1, f"cover_{i+1}")

        # Hub capacity constraints
        for i in range(self.n_nodes):
            model.addCons(quicksum(connection_vars[f"Connection_{i+1}_{j+1}"] for j in range(self.n_nodes)) <= capacities[i] * hub_vars[f"Hub_{i+1}"], f"capacity_{i+1}")

        # Neighborhood connection constraints
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                model.addCons(connection_vars[f"Connection_{i+1}_{j+1}"] <= hub_vars[f"Hub_{i+1}"], f"connect_{i+1}_{j+1}")

        # New network extension/restriction constraints
        for (i, j) in edge_list:
            model.addCons(maintenance_vars[f"Maintenance_{i+1}_{j+1}"] <= self.max_maintenance, f"maintenance_{i+1}_{j+1}")
            model.addCons(maintenance_vars[f"Maintenance_{i+1}_{j+1}"] >= self.min_maintenance, f"min_maintenance_{i+1}_{j+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 15,
        'max_n_nodes': 15,
        'min_n_commodities': 15,
        'max_n_commodities': 22,
        'c_range': (110, 500),
        'd_range': (50, 500),
        'ratio': 500,
        'k_max': 10,
        'er_prob': 0.59,
        'min_hub_cost': 3000,
        'max_hub_cost': 10000,
        'min_connection_cost': 900,
        'max_connection_cost': 937,
        'min_hub_capacity': 100,
        'max_hub_capacity': 3000,
        'min_maintenance_cost': 75,
        'max_maintenance_cost': 2500,
        'max_zones': 45,
        'min_maintenance': 50,
        'max_maintenance': 500,
    }

    drrnetwork = DRRNetwork(parameters, seed=seed)
    instance = drrnetwork.generate_instance()
    solve_status, solve_time = drrnetwork.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")