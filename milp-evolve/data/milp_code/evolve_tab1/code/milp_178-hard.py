import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SupplyChainMILP:
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
    
    def generate_maintenance_data(self):
        maintenance_schedules = {i: np.random.choice([0, 1], size=self.n_time_periods, p=[0.9, 0.1]) for i in range(self.n_nodes)}
        return maintenance_schedules

    def generate_warehouses(self):
        """Generate warehouse capacities and initial stocks."""
        warehouse_caps = np.random.uniform(500, 1500, size=(self.n_warehouses))
        initial_stocks = np.random.uniform(100, 500, size=(self.n_warehouses))
        return warehouse_caps, initial_stocks
    
    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        maintenance_schedules = self.generate_maintenance_data()
        warehouse_caps, initial_stocks = self.generate_warehouses()
        
        res = {
            'commodities': commodities,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'maintenance_schedules': maintenance_schedules,
            'warehouse_caps': warehouse_caps,
            'initial_stocks': initial_stocks
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        maintenance_schedules = instance['maintenance_schedules']
        warehouse_caps = instance['warehouse_caps']
        initial_stocks = instance['initial_stocks']

        model = Model("SupplyChainMILP")

        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        stock_vars = {f"stock_{w+1}_{t+1}": model.addVar(vtype="C", name=f"stock_{w+1}_{t+1}") for w in range(self.n_warehouses) for t in range(self.n_time_periods)}

        # Objective Function: Include transportation, warehousing, and maintenance costs
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            maintenance_schedules[i][t] * stock_vars[f"stock_{i+1}_{t+1}"]
            for i in range(self.n_warehouses) for t in range(self.n_time_periods)
        )

        # Flow Constraints
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Capacity Constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        # Warehouse Capacity Constraints
        for w in range(self.n_warehouses):
            for t in range(self.n_time_periods):
                stock_expr = quicksum(stock_vars[f"stock_{w+1}_{t+1}"] for t in range(self.n_time_periods))
                model.addCons(stock_expr <= warehouse_caps[w], f"warehouse_{w+1}")

        # Symmetry Breaking Constraints
        for w in range(self.n_warehouses - 1):
            model.addCons(stock_vars[f"stock_{w+1}_{1}"] >= stock_vars[f"stock_{w+2}_{1}"], f"symmetry_{w+1}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 20,
        'max_n_nodes': 22,
        'min_n_commodities': 300,
        'max_n_commodities': 315,
        'c_range': (110, 500),
        'd_range': (90, 900),
        'ratio': 1500,
        'k_max': 150,
        'er_prob': 0.24,
        'n_coverage_pairs': 100,
        'coverage_max_cost': 140,
        'n_vehicle_types': 3,
        'max_vehicles_per_type': 15,
        'fuel_range': (2.0, 5.0),
        'emission_range': (0.1, 1.0),
        'delay_range': (1.0, 10.0),
        'max_fuel': 5000,
        'max_emission': 2000,
        'n_time_periods': 5,
        'n_suppliers': 4,
        'n_warehouses': 3  # New for warehouse storage
    }
    
    supply_chain_milp = SupplyChainMILP(parameters, seed=seed)
    instance = supply_chain_milp.generate_instance()
    solve_status, solve_time = supply_chain_milp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")