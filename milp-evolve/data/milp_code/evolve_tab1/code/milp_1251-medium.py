import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ECommerceLogistics:
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

    def generate_orders(self, G):
        orders = []
        for k in range(self.n_orders):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            demand_k = int(np.random.uniform(*self.d_range))
            orders.append((o_k, d_k, demand_k))
        return orders

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_orders = np.random.randint(self.min_n_orders, self.max_n_orders + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        orders = self.generate_orders(G)

        depot_costs = np.random.randint(self.min_depot_cost, self.max_depot_cost + 1, self.n_nodes)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_nodes, self.n_nodes))
        storage_costs = np.random.randint(self.min_storage_cost, self.max_storage_cost + 1, self.n_nodes)
        
        maintenance_costs = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost + 1, (self.n_nodes, self.n_nodes))
        cycles = np.random.randint(0, self.max_cycles, self.n_nodes)

        # New energy consumption costs and capacities:
        energy_costs = np.random.uniform(self.min_energy_cost, self.max_energy_cost, (self.n_nodes, self.n_nodes))
        energy_capacities = np.random.uniform(self.min_energy_capacity, self.max_energy_capacity, (self.n_nodes, self.n_nodes))

        res = {
            'orders': orders, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'depot_costs': depot_costs,
            'delivery_costs': delivery_costs,
            'storage_costs': storage_costs,
            'maintenance_costs': maintenance_costs,
            'cycles': cycles,
            'energy_costs': energy_costs,
            'energy_capacities': energy_capacities,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        orders = instance['orders']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        
        depot_costs = instance['depot_costs']
        delivery_costs = instance['delivery_costs']
        storage_costs = instance['storage_costs']
        maintenance_costs = instance['maintenance_costs']
        cycles = instance['cycles']
        energy_costs = instance['energy_costs']
        energy_capacities = instance['energy_capacities']

        model = Model("ECommerceLogistics")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_orders)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        hub_vars = {f"Hub_{i+1}": model.addVar(vtype="B", name=f"Hub_{i+1}") for i in range(self.n_nodes)}
        connection_vars = {f"Connection_{i+1}_{j+1}": model.addVar(vtype="B", name=f"Connection_{i+1}_{j+1}") for i in range(self.n_nodes) for j in range(self.n_nodes)}
        
        # New highway load variables
        highway_load_vars = {f"HighwayLoad_{i+1}_{j+1}": model.addVar(vtype="C", name=f"HighwayLoad_{i+1}_{j+1}") for (i, j) in edge_list}
        # New new node construction variables
        new_node_vars = {f"NewNode_{i+1}_{j+1}": model.addVar(vtype="B", name=f"NewNode_{i+1}_{j+1}") for (i, j) in edge_list}

        # Update objective function to include new costs
        objective_expr = quicksum(
            orders[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_orders)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            depot_costs[i] * hub_vars[f"Hub_{i+1}"]
            for i in range(self.n_nodes)
        )
        objective_expr += quicksum(
            delivery_costs[i, j] * connection_vars[f"Connection_{i+1}_{j+1}"]
            for i in range(self.n_nodes) for j in range(self.n_nodes)
        )
        objective_expr += quicksum(
            maintenance_costs[i, j] * highway_load_vars[f"HighwayLoad_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        # Including energy costs
        objective_expr += quicksum(
            energy_costs[i, j] * new_node_vars[f"NewNode_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )

        for i in range(self.n_nodes):
            for k in range(self.n_orders):
                delta_i = 1 if orders[k][0] == i else -1 if orders[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(orders[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_orders))
            M = sum(orders[k][2] for k in range(self.n_orders))  # Big M
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")
            model.addCons(arc_expr <= M * y_vars[f"y_{i+1}_{j+1}"], f"big_M_arc_{i+1}_{j+1}")

        # Set covering constraints: Ensure that every node is covered
        for i in range(self.n_nodes):
            cover_expr_in = quicksum(y_vars[f"y_{j+1}_{i+1}"] for j in incommings[i])
            cover_expr_out = quicksum(y_vars[f"y_{i+1}_{j+1}"] for j in outcommings[i])
            model.addCons(cover_expr_in + cover_expr_out >= 1, f"cover_{i+1}")

        # Hub capacity constraints
        for i in range(self.n_nodes):
            model.addCons(quicksum(connection_vars[f"Connection_{i+1}_{j+1}"] for j in range(self.n_nodes)) <= storage_costs[i] * hub_vars[f"Hub_{i+1}"], f"capacity_{i+1}")

        # Neighborhood connection constraints
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                model.addCons(connection_vars[f"Connection_{i+1}_{j+1}"] <= hub_vars[f"Hub_{i+1}"], f"connect_{i+1}_{j+1}")

        # New node installation/extension limitations
        for (i, j) in edge_list:
            model.addCons(highway_load_vars[f"HighwayLoad_{i+1}_{j+1}"] <= self.max_highway_load, f"highway_load_{i+1}_{j+1}")
            model.addCons(highway_load_vars[f"HighwayLoad_{i+1}_{j+1}"] >= self.min_highway_load, f"min_highway_load_{i+1}_{j+1}")
            # Energy constraints:
            model.addCons(new_node_vars[f"NewNode_{i+1}_{j+1}"] <= energy_capacities[i, j], f"energy_capacity_{i+1}_{j+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 30,
        'max_n_nodes': 30,
        'min_n_orders': 30,
        'max_n_orders': 154,
        'c_range': (110, 500),
        'd_range': (100, 1000),
        'ratio': 250,
        'k_max': 100,
        'er_prob': 0.31,
        'min_depot_cost': 3000,
        'max_depot_cost': 10000,
        'min_delivery_cost': 450,
        'max_delivery_cost': 2811,
        'min_storage_cost': 100,
        'max_storage_cost': 3000,
        'min_maintenance_cost': 750,
        'max_maintenance_cost': 1250,
        'max_cycles': 22,
        'min_highway_load': 350,
        'max_highway_load': 2500,
        'min_energy_cost': 900,
        'max_energy_cost': 2000,
        'min_energy_capacity': 250,
        'max_energy_capacity': 1000,
    }

    logistics = ECommerceLogistics(parameters, seed=seed)
    instance = logistics.generate_instance()
    solve_status, solve_time = logistics.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")