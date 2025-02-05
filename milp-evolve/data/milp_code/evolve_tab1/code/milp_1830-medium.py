import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EVCSNO:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_stations, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_stations, self.n_stations), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_stations)}
        outcommings = {i: [] for i in range(self.n_stations)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_vehicles(self, G):
        vehicles = []
        for k in range(self.n_vehicles):
            while True:
                o_k = np.random.randint(0, self.n_stations)
                d_k = np.random.randint(0, self.n_stations)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            # integer demands
            demand_k = int(np.random.uniform(*self.d_range))
            vehicles.append((o_k, d_k, demand_k))
        return vehicles

    def generate_instance(self):
        self.n_stations = np.random.randint(self.min_n_stations, self.max_n_stations+1)
        self.n_vehicles = np.random.randint(self.min_n_vehicles, self.max_n_vehicles + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        vehicles = self.generate_vehicles(G)

        res = {
            'vehicles': vehicles, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        vehicles = instance['vehicles']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        
        model = Model("EVCSNO")
        ElectricVehicleChargingXVars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_vehicles)}
        BatteryCapacityBVars = {f"b_{i+1}_{j+1}": model.addVar(vtype="B", name=f"b_{i+1}_{j+1}") for (i, j) in edge_list}

        objective_expr = quicksum(
            vehicles[k][2] * adj_mat[i, j][0] * ElectricVehicleChargingXVars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_vehicles)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * BatteryCapacityBVars[f"b_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )

        for i in range(self.n_stations):
            for k in range(self.n_vehicles):
                delta_i = 1 if vehicles[k][0] == i else -1 if vehicles[k][1] == i else 0
                flow_expr = quicksum(ElectricVehicleChargingXVars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(ElectricVehicleChargingXVars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"power_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(vehicles[k][2] * ElectricVehicleChargingXVars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_vehicles))
            model.addCons(arc_expr <= adj_mat[i, j][2] * BatteryCapacityBVars[f"b_{i+1}_{j+1}"], f"capacity_{i+1}_{j+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_stations': 20,
        'max_n_stations': 30,
        'min_n_vehicles': 15,
        'max_n_vehicles': 22,
        'c_range': (11, 50),
        'd_range': (5, 50),
        'ratio': 50,
        'k_max': 70,
        'er_prob': 0.38,
    }

    evcsno = EVCSNO(parameters, seed=seed)
    instance = evcsno.generate_instance()
    solve_status, solve_time = evcsno.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")