import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedFCMCNF:
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
    
    def generate_pollution_data(self):
        pollution_levels = {i: np.random.uniform(self.pollution_range[0], self.pollution_range[1]) for i in range(self.n_nodes)}
        return pollution_levels

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        maintenance_schedules = self.generate_maintenance_data()
        pollution_levels = self.generate_pollution_data()
        
        res = {
            'commodities': commodities,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'maintenance_schedules': maintenance_schedules,
            'pollution_levels': pollution_levels,
        }

        # Example of adding random critical edges
        self.critical_edges = random.sample(edge_list, min(10, len(edge_list)))
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        maintenance_schedules = instance['maintenance_schedules']
        pollution_levels = instance['pollution_levels']

        model = Model("SimplifiedFCMCNF")

        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        m_vars = {f"m_{i+1}_{t+1}": model.addVar(vtype="B", name=f"m_{i+1}_{t+1}") for i in range(self.n_nodes) for t in range(self.n_time_periods)}
        
        # New Variable for Pollution Constraints
        p_vars = {f"p_{i+1}": model.addVar(vtype="B", name=f"p_{i+1}") for i in range(self.n_nodes)}

        # New Variables and Logical Conditions for critical edges
        z_vars = {f"z_{i+1}_{j+1}": model.addVar(vtype="B", name=f"z_{i+1}_{j+1}") for (i, j) in self.critical_edges}

        # Objective Function: Include maintenance and fluctuation costs
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            maintenance_schedules[i][t] * m_vars[f"m_{i+1}_{t+1}"]
            for i in range(self.n_nodes) for t in range(self.n_time_periods)
        )
        
        # Add penalties for pollution levels
        objective_expr += quicksum(
            pollution_levels[i] * p_vars[f"p_{i+1}"]
            for i in range(self.n_nodes)
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

        # Maintenance Constraints
        for i in range(self.n_nodes):
            for t in range(self.n_time_periods):
                maintain_expr = quicksum(m_vars[f"m_{i+1}_{t+1}"] for i in range(self.n_nodes)) 
                model.addCons(maintain_expr <= 1, f"maintain_{i+1}_{t+1}")
                
        # Logical Constraints for critical edges
        for (i, j) in self.critical_edges:
            critical_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            alt_routes_expr = quicksum(x_vars[f"x_{p+1}_{q+1}_{k+1}"] for (p, q) in edge_list if (p, q) != (i, j) for k in range(self.n_commodities))
            model.addCons(critical_expr <= z_vars[f"z_{i+1}_{j+1}"] * 1e6)
            model.addCons(alt_routes_expr >= z_vars[f"z_{i+1}_{j+1}"])

        # If a node is under maintenance, adjacent edges have zero flow
        for i in range(self.n_nodes):
            for t in range(self.n_time_periods):
                for j in outcommings[i]:
                    model.addCons(m_vars[f"m_{i+1}_{t+1}"] + x_vars[f"x_{i+1}_{j+1}_1"] <= 1)
                for j in incommings[i]:
                    model.addCons(m_vars[f"m_{i+1}_{t+1}"] + x_vars[f"x_{j+1}_{i+1}_1"] <= 1)

        # Air quality constraints
        total_pollution_expr = quicksum(
            pollution_levels[i] * p_vars[f"p_{i+1}"]
            for i in range(self.n_nodes)
        )
        model.addCons(total_pollution_expr <= self.max_pollution, "pollution_limit")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 10,
        'max_n_nodes': 11,
        'min_n_commodities': 150,
        'max_n_commodities': 785,
        'c_range': (410, 1875),
        'd_range': (180, 1800),
        'ratio': 3000,
        'k_max': 84,
        'er_prob': 0.8,
        'n_coverage_pairs': 1500,
        'coverage_max_cost': 2940,
        'n_vehicle_types': 243,
        'max_vehicles_per_type': 105,
        'fuel_range': (13.5, 33.75),
        'emission_range': (0.1, 0.1),
        'delay_range': (21.0, 210.0),
        'max_fuel': 5000,
        'max_emission': 500,
        'n_time_periods': 15,
        'n_suppliers': 180,
        'pollution_range': (90, 900),
        'max_pollution': 3000,
    }
    
    fcmcnf = SimplifiedFCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")