import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FCMCNF_Simplified:
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
            # integer demands
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        # Additional data for carbon emissions and SLA
        carbon_emissions = {(i, j): np.random.uniform(*self.carbon_range) for (i, j) in edge_list}
        sla_constraints = [np.random.randint(1, self.max_sla) for k in range(self.n_commodities)]
        
        # Representation of major highways and ports
        highways_ports_proximity = np.random.randint(0, 2, self.n_nodes).tolist()

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'carbon_emissions': carbon_emissions,
            'sla_constraints': sla_constraints,
            'highways_ports_proximity': highways_ports_proximity
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        carbon_emissions = instance['carbon_emissions']
        sla_constraints = instance['sla_constraints']
        highways_ports_proximity = instance['highways_ports_proximity']
        
        model = Model("FCMCNF_Simplified")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        w_vars = {f"w_{i+1}": model.addVar(vtype="B", name=f"w_{i+1}") for i in range(self.n_nodes)}

        # Objective function
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        
        # Minimize warehouse locations cost
        objective_expr += quicksum(
            self.warehouse_location_cost * w_vars[f"w_{i+1}"]
            for i in range(self.n_nodes)
        )

        # Set Objective
        model.setObjective(objective_expr, "minimize")

        # Flow conservation constraints
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Capacity constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        # Carbon emission constraints
        total_carbon_emission = quicksum(
            carbon_emissions[(i, j)] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        model.addCons(total_carbon_emission <= self.carbon_limit, "carbon_limit")

        # Service Level Agreement (SLA) constraints
        for k in range(self.n_commodities):
            for (i, j) in edge_list:
                travel_time = adj_mat[i, j][0] / self.speed
                model.addCons(travel_time * x_vars[f"x_{i+1}_{j+1}_{k+1}"] <= sla_constraints[k], f"sla_{k+1}_{i+1}_{j+1}")
        
        # Proximity to highways and ports constraint
        for i in range(self.n_nodes):
            if highways_ports_proximity[i] == 0:
                model.addCons(w_vars[f"w_{i+1}"] == 0, f"proximity_{i+1}")

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
        'carbon_limit': 5000,
        'carbon_range': (1, 20),
        'max_sla': 50,
        'speed': 60,
        'warehouse_location_cost': 1000,
    }

    fcmcnf_simplified = FCMCNF_Simplified(parameters, seed=seed)
    instance = fcmcnf_simplified.generate_instance()
    solve_status, solve_time = fcmcnf_simplified.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")