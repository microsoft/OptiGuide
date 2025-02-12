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

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        # Generate data for facility location
        self.n_facilities = self.n_nodes
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        
        # Generate data for emergency supply distribution
        supply_opening_costs = np.random.randint(self.min_supply_cost, self.max_supply_cost + 1, self.n_facilities)
        transport_times = np.random.uniform(self.min_transport_time, self.max_transport_time, (self.n_facilities, self.max_n_commodities))

        # Generate time windows for deliveries
        delivery_windows = {k: (np.random.uniform(0, self.max_time_window_start), np.random.uniform(self.min_time_window_end, self.max_time_window_end)) for k in range(self.n_commodities)}

        # Generate charging data for electric vehicles
        charging_times = np.random.uniform(self.min_charging_time, self.max_charging_time, self.n_facilities)
        charging_station_caps = np.random.randint(self.min_charging_station_cap, self.max_charging_station_cap + 1, self.n_facilities)

        # Assign priorities to customers
        customer_priorities = np.random.randint(1, self.max_priority + 1, self.n_commodities)

        # Generate traffic congestion effects
        congestion_factors = np.random.uniform(self.min_congestion_factor, self.max_congestion_factor, (self.n_nodes, self.n_nodes))

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'facility_costs': facility_costs,
            'supply_opening_costs': supply_opening_costs,
            'transport_times': transport_times,
            'delivery_windows': delivery_windows,
            'charging_times': charging_times,
            'charging_station_caps': charging_station_caps,
            'customer_priorities': customer_priorities,
            'congestion_factors': congestion_factors,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        facility_costs = instance['facility_costs']
        supply_opening_costs = instance['supply_opening_costs']
        transport_times = instance['transport_times']
        delivery_windows = instance['delivery_windows']
        charging_times = instance['charging_times']
        charging_station_caps = instance['charging_station_caps']
        customer_priorities = instance['customer_priorities']
        congestion_factors = instance['congestion_factors']

        model = Model("FCMCNF")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}

        open_vars = {i: model.addVar(vtype="B", name=f"Facility_{i+1}") for i in range(self.n_nodes)}

        supply_open_vars = {s: model.addVar(vtype="B", name=f"Supply_{s}") for s in range(self.n_nodes)}
        supply_assignment_vars = {(s, k): model.addVar(vtype="I", lb=0, name=f"Supply_{s}_Commodity_{k}") for s in range(self.n_nodes) for k in range(self.n_commodities)}

        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr -= quicksum(
            facility_costs[i] * open_vars[i]
            for i in range(self.n_nodes)
        )

        # New addition to the objective for emergency supply distribution costs
        objective_expr += quicksum(
            supply_opening_costs[s] * supply_open_vars[s]
            for s in range(self.n_nodes)
        )
        objective_expr += quicksum(
            transport_times[s, k] * supply_assignment_vars[s, k]
            for s in range(self.n_nodes) for k in range(self.n_commodities)
        )

        # Adding priority penalties
        objective_expr += quicksum(
            customer_priorities[k] * max(0, transport_times[s, k] - delivery_windows[k][1]) * supply_assignment_vars[s,k]
            for s in range(self.n_nodes) for k in range(self.n_commodities)
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        for s in range(self.n_nodes):
            for k in range(self.n_commodities):
                supply_flow_expr = quicksum(x_vars[f"x_{s+1}_{j+1}_{k+1}"] for j in outcommings[s])
                model.addCons(supply_flow_expr >= supply_assignment_vars[s, k], f"supply_flow_{s+1}_{k+1}")

        # EV charging constraints
        for s in range(self.n_nodes):
            model.addCons(quicksum(supply_assignment_vars[s, k] for k in range(self.n_commodities)) <= charging_station_caps[s], f'charging_cap_{s}')

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 11,
        'max_n_nodes': 15,
        'min_n_commodities': 88,
        'max_n_commodities': 123,
        'c_range': (175, 875),
        'd_range': (12, 125),
        'ratio': 506,
        'k_max': 1050,
        'er_prob': 0.24,
        'min_facility_cost': 1000,
        'max_facility_cost': 2500,
        'min_supply_cost': 3000,
        'max_supply_cost': 5000,
        'min_transport_time': 0.1,
        'max_transport_time': 187.5,
        'max_time_window_start': 750,
        'min_time_window_end': 300,
        'max_time_window_end': 700,
        'min_charging_time': 20,
        'max_charging_time': 600,
        'min_charging_station_cap': 4,
        'max_charging_station_cap': 5,
        'max_priority': 10,
        'min_congestion_factor': 0,
        'max_congestion_factor': 0.75,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")