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
        coverage_costs = np.random.randint(1, self.coverage_max_cost + 1, size=self.n_nodes)
        coverage_pairs = [(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes) if i != j]
        chosen_pairs = np.random.choice(len(coverage_pairs), size=self.n_coverage_pairs, replace=False)
        coverage_set = [coverage_pairs[i] for i in chosen_pairs]
        return coverage_costs, coverage_set
        
    def generate_facility_data(self):
        facility_costs = np.random.randint(1, self.facility_max_cost + 1, size=self.n_nodes)
        facility_caps = np.random.randint(1, self.facility_max_cap + 1, size=self.n_nodes)
        return facility_costs, facility_caps

    def generate_labor_shifts(self):
        shifts = np.random.randint(1, self.max_shifts + 1)
        shift_capacity = np.random.randint(1, self.max_shift_capacity + 1, size=shifts)
        return shifts, shift_capacity

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        coverage_costs, coverage_set = self.generate_coverage_data()
        facility_costs, facility_caps = self.generate_facility_data()
        shifts, shift_capacity = self.generate_labor_shifts()
        maintenance_thresh = np.random.randint(self.min_maintenance_thresh, self.max_maintenance_thresh)

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings, 
            'coverage_costs': coverage_costs, 
            'coverage_set': coverage_set,
            'facility_costs': facility_costs,
            'facility_caps': facility_caps,
            'shifts': shifts,
            'shift_capacity': shift_capacity,
            'maintenance_thresh': maintenance_thresh
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
        facility_costs = instance['facility_costs']
        facility_caps = instance['facility_caps']
        shifts = instance['shifts']
        shift_capacity = instance['shift_capacity']
        maintenance_thresh = instance['maintenance_thresh']
        
        model = Model("FCMCNF")
        x_vars = {f"x_{i + 1}_{j + 1}_{k + 1}": model.addVar(vtype="C", name=f"x_{i + 1}_{j + 1}_{k + 1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i + 1}_{j + 1}": model.addVar(vtype="B", name=f"y_{i + 1}_{j + 1}") for (i, j) in edge_list}
        z_vars = {f"z_{i + 1}_{j + 1}": model.addVar(vtype="B", name=f"z_{i + 1}_{j + 1}") for (i, j) in coverage_set}
        w_vars = {f"w_{i + 1}": model.addVar(vtype="B", name=f"w_{i + 1}") for i in range(self.n_nodes)}

        # New variables
        shift_vars = {f"shift_{m + 1}_{s + 1}": model.addVar(vtype="B", name=f"shift_{m + 1}_{s + 1}") for m in range(self.n_nodes) for s in range(shifts)}
        usage_vars = {f"usage_{m + 1}_{t + 1}": model.addVar(vtype="C", name=f"usage_{m + 1}_{t + 1}") for m in range(self.n_nodes) for t in range(self.n_nodes)}
        maintenance_vars = {f"maintenance_{m + 1}": model.addVar(vtype="B", name=f"maintenance_{m + 1}") for m in range(self.n_nodes)}
        rush_vars = {f"rush_{k + 1}": model.addVar(vtype="B", name=f"rush_{k + 1}") for k in range(self.n_commodities)}

        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i + 1}_{j + 1}_{k + 1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i + 1}_{j + 1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            coverage_costs[i] * z_vars[f"z_{i + 1}_{j + 1}"]
            for (i, j) in coverage_set
        )
        objective_expr += quicksum(
            facility_costs[i] * w_vars[f"w_{i + 1}"]
            for i in range(self.n_nodes)
        )

        # New objectives
        objective_expr += quicksum(
            shift_capacity[s] * shift_vars[f"shift_{m + 1}_{s + 1}"]
            for m in range(self.n_nodes) for s in range(shifts)
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i + 1}_{j + 1}_{k + 1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j + 1}_{i + 1}_{k + 1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i + 1}_{k + 1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i + 1}_{j + 1}_{k + 1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i + 1}_{j + 1}"], f"arc_{i + 1}_{j + 1}")

        for i in range(self.n_nodes):
            coverage_expr = quicksum(z_vars[f"z_{j + 1}_{i + 1}"] for j in range(self.n_nodes) if (j, i) in coverage_set)
            model.addCons(coverage_expr >= 1, f"coverage_{i + 1}")

        for i in range(self.n_nodes):
            facility_capacity_expr = quicksum(commodities[k][2] * x_vars[f"x_{i + 1}_{j + 1}_{k + 1}"] for j in outcommings[i] for k in range(self.n_commodities)) 
            model.addCons(facility_capacity_expr <= facility_caps[i] + w_vars[f'w_{i + 1}'] * sum(commodities[k][2] for k in range(self.n_commodities)), f"facility_{i + 1}")

        # New constraints for shift scheduling
        for m in range(self.n_nodes):
            model.addCons(quicksum(shift_vars[f"shift_{m + 1}_{s + 1}"] for s in range(shifts)) <= 1, f"machine_shift_{m + 1}")

        for s in range(shifts):
            shift_labor_expr = quicksum(shift_vars[f"shift_{m + 1}_{s + 1}"] for m in range(self.n_nodes))
            model.addCons(shift_labor_expr <= shift_capacity[s], f"shift_labor_{s + 1}")

        # New constraints for machine maintenance
        for m in range(self.n_nodes):
            cum_usage_expr = quicksum(usage_vars[f"usage_{m + 1}_{t + 1}"] for t in range(self.n_nodes))
            model.addCons(cum_usage_expr <= maintenance_thresh * (1 - maintenance_vars[f"maintenance_{m + 1}"]), f"maintenance_thresh_{m + 1}")

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
        'facility_max_cost': 30,
        'facility_max_cap': 200,
        'min_maintenance_thresh': 30,
        'max_maintenance_thresh': 50,
        'max_shifts': 3,
        'max_shift_capacity': 5
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")