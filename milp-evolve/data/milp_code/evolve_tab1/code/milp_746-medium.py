import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FCMCNF_Complex:
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

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings
        }
        
        # Adding additional data for maintenance scheduling, energy consumption, and carbon emissions
        maintenance_schedule = {
            node: sorted(random.sample(range(self.horizon), self.maintenance_slots))
            for node in range(self.n_nodes)
        }
        energy_consumption = {edge: np.random.uniform(1.0, 5.0) for edge in edge_list}
        carbon_cost_coefficients = {edge: np.random.uniform(0.5, 2.0) for edge in edge_list}

        res.update({
            'maintenance_schedule': maintenance_schedule,
            'energy_consumption': energy_consumption,
            'carbon_cost_coefficients': carbon_cost_coefficients,
        })

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        maintenance_schedule = instance['maintenance_schedule']
        energy_consumption = instance['energy_consumption']
        carbon_cost_coefficients = instance['carbon_cost_coefficients']
        
        model = Model("FCMCNF_Complex")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        
        # New variables for maintenance, energy consumption, and carbon emissions
        maintenance_vars = {f"m_{i+1}_{t}": model.addVar(vtype="B", name=f"m_{i+1}_{t}") for i in range(self.n_nodes) for t in range(self.horizon)}
        carbon_emissions_vars = {f"c_{i+1}_{j+1}": model.addVar(vtype="C", name=f"c_{i+1}_{j+1}") for (i, j) in edge_list}

        # Objective function
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        # Adding carbon emissions cost to objective
        objective_expr += quicksum(
            carbon_cost_coefficients[(i, j)] * carbon_emissions_vars[f"c_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
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

        # Additional constraints for maintenance scheduling and carbon emissions
        for i in range(self.n_nodes):
            for t in range(self.horizon):
                if t in maintenance_schedule[i]:
                    model.addCons(maintenance_vars[f"m_{i+1}_{t}"] == 1, f"maintenance_{i+1}_{t}")
                else:
                    model.addCons(maintenance_vars[f"m_{i+1}_{t}"] == 0, f"maintenance_not_{i+1}_{t}")

        # Carbon emissions constraints
        for (i, j) in edge_list:
            model.addCons(
                carbon_emissions_vars[f"c_{i+1}_{j+1}"] == quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] * carbon_cost_coefficients[(i, j)] for k in range(self.n_commodities)),
                f"carbon_emission_{i+1}_{j+1}"
            )
        
        model.addCons(
            quicksum(carbon_emissions_vars[f"c_{i+1}_{j+1}"] for (i, j) in edge_list) <= self.sustainability_budget,
            "total_carbon_emissions_limit"
        )
        
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
        'horizon': 18,
        'maintenance_slots': 6,
        'sustainability_budget': 5000,  # Parameter for carbon emissions budget
    }

    fcmcnf_complex = FCMCNF_Complex(parameters, seed=seed)
    instance = fcmcnf_complex.generate_instance()
    solve_status, solve_time = fcmcnf_complex.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")