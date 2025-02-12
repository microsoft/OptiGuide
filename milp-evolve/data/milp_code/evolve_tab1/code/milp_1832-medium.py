import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HybridOptimization:
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

    def truck_travel_times(self):
        base_travel_time = 20.0  # base travel time in minutes
        return base_travel_time * np.random.rand(self.n_trucks, self.n_nodes)
        
    def generate_instance(self):
        # Nodes, commodities, and arc properties from FCMCNF
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        # New instance data from UrbanLogistics
        truck_arrival_rates = np.random.randint(*self.arrival_rate_interval, self.n_trucks)
        center_capacities = np.random.randint(*self.capacity_interval, self.n_nodes)
        activation_costs = np.random.randint(*self.activation_cost_interval, self.n_nodes)
        truck_travel_times = self.truck_travel_times()
        time_windows = np.random.randint(*self.delivery_window_interval, self.n_trucks)

        center_capacities = center_capacities * self.ratio * np.sum(truck_arrival_rates) / np.sum(center_capacities)
        center_capacities = np.round(center_capacities)
        
        return {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'truck_arrival_rates': truck_arrival_rates,
            'center_capacities': center_capacities,
            'activation_costs': activation_costs,
            'truck_travel_times': truck_travel_times,
            'time_windows': time_windows
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        truck_arrival_rates = instance['truck_arrival_rates']
        center_capacities = instance['center_capacities']
        activation_costs = instance['activation_costs']
        truck_travel_times = instance['truck_travel_times']
        time_windows = instance['time_windows']

        model = Model("HybridOptimization")

        # FCMCNF variables
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}

        # UrbanLogistics variables
        activate_center = {m: model.addVar(vtype="B", name=f"Activate_{m}") for m in range(self.n_nodes)}
        allocate_truck = {(n, m): model.addVar(vtype="B", name=f"Allocate_{n}_{m}") for n in range(self.n_trucks) for m in range(self.n_nodes)}
        delivery_time = {(n, m): model.addVar(vtype="C", name=f"DeliveryTime_{n}_{m}") for n in range(self.n_trucks) for m in range(self.n_nodes)}

        # Objective with mixed focuses
        penalty_per_travel_time = 100
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(activation_costs[m] * activate_center[m] for m in range(self.n_nodes)) + \
                         penalty_per_travel_time * quicksum(truck_travel_times[n, m] * allocate_truck[n, m] for n in range(self.n_trucks) for m in range(self.n_nodes))

        # Flow constraints
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        # UrbanLogistics constraints
        for n in range(self.n_trucks):
            model.addCons(quicksum(allocate_truck[n, m] for m in range(self.n_nodes)) == 1, f"Truck_Allocation_{n}")
        
        for m in range(self.n_nodes):
            model.addCons(quicksum(truck_arrival_rates[n] * allocate_truck[n, m] for n in range(self.n_trucks)) <= center_capacities[m] * activate_center[m], f"Center_Capacity_{m}")
        
        for n in range(self.n_trucks):
            for m in range(self.n_nodes):
                model.addCons(truck_travel_times[n, m] * allocate_truck[n, m] <= activate_center[m] * 180, f"Travel_Time_Limit_{n}_{m}")

        for n in range(self.n_trucks):
            for m in range(self.n_nodes):
                model.addCons(delivery_time[n, m] == time_windows[n], f"Delivery_Time_Window_{n}_{m}")

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
        'min_n_commodities': 1,
        'max_n_commodities': 45,
        'c_range': (110, 500),
        'd_range': (100, 1000),
        'ratio': 50,
        'k_max': 100,
        'er_prob': 0.38,
        'n_trucks': 100,
        'arrival_rate_interval': (22, 225),
        'capacity_interval': (33, 101),
        'activation_cost_interval': (1125, 2250),
        'delivery_window_interval': (405, 1620),
    }

    hybrid_optimization = HybridOptimization(parameters, seed=seed)
    instance = hybrid_optimization.generate_instance()
    solve_status, solve_time = hybrid_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")