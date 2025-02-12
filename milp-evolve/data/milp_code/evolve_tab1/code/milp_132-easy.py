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

    def generate_vehicle_data(self):
        vehicle_types = range(self.n_vehicle_types)
        vehicle_avail = {v: np.random.randint(1, self.max_vehicles_per_type + 1) for v in vehicle_types}
        fuel_consumption = {v: np.random.uniform(*self.fuel_range) for v in vehicle_types}
        emission_factors = {v: np.random.uniform(*self.emission_range) for v in vehicle_types}
        delay_factors = np.random.uniform(*self.delay_range, size=(self.n_nodes, self.n_nodes))
        return vehicle_types, vehicle_avail, fuel_consumption, emission_factors, delay_factors

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        coverage_costs, coverage_set = self.generate_coverage_data()
        vehicle_types, vehicle_avail, fuel_consumption, emission_factors, delay_factors = self.generate_vehicle_data()

        res = {
            'commodities': commodities,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'coverage_costs': coverage_costs,
            'coverage_set': coverage_set,
            'vehicle_types': vehicle_types,
            'vehicle_avail': vehicle_avail,
            'fuel_consumption': fuel_consumption,
            'emission_factors': emission_factors,
            'delay_factors': delay_factors
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
        vehicle_types = instance['vehicle_types']
        vehicle_avail = instance['vehicle_avail']
        fuel_consumption = instance['fuel_consumption']
        emission_factors = instance['emission_factors']
        delay_factors = instance['delay_factors']

        model = Model("FCMCNF")

        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        z_vars = {f"z_{i+1}_{j+1}": model.addVar(vtype="B", name=f"z_{i+1}_{j+1}") for (i, j) in coverage_set}
        v_vars = {f"v_{i+1}_{j+1}_{v+1}": model.addVar(vtype="B", name=f"v_{i+1}_{j+1}_{v+1}") for (i, j) in edge_list for v in vehicle_types}

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
            delay_factors[i, j] * quicksum(v_vars[f"v_{i+1}_{j+1}_{v+1}"] for v in vehicle_types)
            for (i, j) in edge_list
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")
        
        for v in vehicle_types:
            vehicle_avail_expr = quicksum(v_vars[f"v_{i+1}_{j+1}_{v+1}"] for (i, j) in edge_list)
            model.addCons(vehicle_avail_expr <= vehicle_avail[v], f"vehicle_avail_{v+1}")

        for i in range(self.n_nodes):
            coverage_expr = quicksum(z_vars[f"z_{j+1}_{i+1}"] for j in range(self.n_nodes) if (j, i) in coverage_set)
            model.addCons(coverage_expr >= 1, f"coverage_{i+1}")

        fuel_expr = quicksum(
            adj_mat[i,j][0] * fuel_consumption[v] * v_vars[f"v_{i+1}_{j+1}_{v+1}"]
            for (i, j) in edge_list for v in vehicle_types
        )
        model.addCons(fuel_expr <= self.max_fuel, f"fuel")

        emission_expr = quicksum(
            adj_mat[i, j][0] * emission_factors[v] * v_vars[f"v_{i+1}_{j+1}_{v+1}"]
            for (i, j) in edge_list for v in vehicle_types
        )
        model.addCons(emission_expr <= self.max_emission, f"emission")

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
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")