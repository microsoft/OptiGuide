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
        G = nx.erdos_renyi_graph(
            n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True
        )
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            energy_ij = np.random.uniform(*self.energy_range)
            emissions_ij = np.random.uniform(*self.emissions_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij, energy_ij, emissions_ij)
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

    def generate_labor_schedule(self, n_nodes):
        labor_schedule = {
            node: int(np.random.choice([0, 1], p=[self.labor_unavailability_prob, 1 - self.labor_unavailability_prob]))
            for node in range(n_nodes)
        }
        return labor_schedule

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        labor_schedule = self.generate_labor_schedule(self.n_nodes)

        # Generate mutual exclusivity pairs
        self.n_exclusive_pairs = min(self.n_exclusive_pairs, len(edge_list) // 2)
        mutual_exclusivity_pairs = [(random.choice(edge_list), random.choice(edge_list)) for _ in range(self.n_exclusive_pairs)]
        
        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'mutual_exclusivity_pairs': mutual_exclusivity_pairs,
            'energy_threshold': self.energy_threshold,
            'emissions_threshold': self.emissions_threshold,
            'labor_schedule': labor_schedule
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        energy_threshold = instance['energy_threshold']
        emissions_threshold = instance['emissions_threshold']
        labor_schedule = instance['labor_schedule']

        model = Model("FCMCNF")
        x_vars = {
            f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}")
            for (i, j) in edge_list for k in range(self.n_commodities)
        }
        y_vars = {
            f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}")
            for (i, j) in edge_list
        }
        waste_vars = {
            f"waste_{i+1}_{j+1}": model.addVar(vtype="C", name=f"waste_{i+1}_{j+1}")
            for (i, j) in edge_list
        }
        labor_vars = {
            f"labor_{i+1}": model.addVar(vtype="B", name=f"labor_{i+1}")
            for i in range(self.n_nodes)
        }
        overtime_vars = {
            f"overtime_{i+1}": model.addVar(vtype="B", name=f"overtime_{i+1}")
            for i in range(self.n_nodes)
        }

        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"] for (i, j) in edge_list
        )
        objective_expr += self.waste_penalty * quicksum(
            waste_vars[f"waste_{i+1}_{j+1}"] for (i, j) in edge_list
        )
        objective_expr += self.penalty_overtime * quicksum(
            overtime_vars[f"overtime_{i+1}"] for i in range(self.n_nodes)
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                # 1 if source, -1 if sink, 0 if else
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(
                    x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]
                ) - quicksum(
                    x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i]
                )
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(
                commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
                for k in range(self.n_commodities)
            )
            model.addCons(
                arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}"
            )

        # Environmental constraints
        energy_consumed = quicksum(
            adj_mat[i, j][3] * y_vars[f"y_{i+1}_{j+1}"] for (i, j) in edge_list
        )
        emissions_produced = quicksum(
            adj_mat[i, j][4] * y_vars[f"y_{i+1}_{j+1}"] for (i, j) in edge_list
        )
        model.addCons(energy_consumed <= energy_threshold, "energy_limit")
        model.addCons(emissions_produced <= emissions_threshold, "emissions_limit")

        # Mutual exclusivity constraints
        for (edge1, edge2) in mutual_exclusivity_pairs:
            i1, j1 = edge1
            i2, j2 = edge2
            if (i1, j1) in edge_list and (i2, j2) in edge_list:
                model.addCons(
                    y_vars[f"y_{i1+1}_{j1+1}"] + y_vars[f"y_{i2+1}_{j2+1}"] <= 1, f"mutual_exclusivity_{i1+1}_{j1+1}_{i2+1}_{j2+1}"
                )

        # Waste constraints
        for (i, j) in edge_list:
            model.addCons(waste_vars[f"waste_{i+1}_{j+1}"] >= 0, f"waste_LB_{i+1}_{j+1}")
            model.addCons(waste_vars[f"waste_{i+1}_{j+1}"] >= self.waste_factor * y_vars[f"y_{i+1}_{j+1}"], f"waste_link_{i+1}_{j+1}")

        # Labor Availability Constraints
        for i in range(self.n_nodes):
            model.addCons(labor_vars[f"labor_{i+1}"] == 1 - labor_schedule[i], f"labor_availability_{i+1}")

        # Overtime penalty constraints
        for i in range(self.n_nodes):
            model.addCons(quicksum(y_vars[f"y_{i+1}_{j+1}"] for j in outcommings[i]) <= labor_vars[f"labor_{i+1}"] * self.labor_capacity + overtime_vars[f"overtime_{i+1}"] * self.overtime_capacity, f"node_overtime_{i+1}")

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
        'min_n_commodities': 7,
        'max_n_commodities': 11,
        'c_range': (11, 50),
        'd_range': (0, 5),
        'ratio': 50,
        'k_max': 5,
        'er_prob': 0.66,
        'energy_range': (0, 0),
        'emissions_range': (0, 5),
        'energy_threshold': 100,
        'emissions_threshold': 100,
        'waste_penalty': 0.52,
        'waste_factor': 0.45,
        'n_exclusive_pairs': 100,
        'penalty_overtime': 0.2,
        'labor_unavailability_prob': 0.52,
        'labor_capacity': 1,
        'overtime_capacity': 9,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")