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

    def generate_barabasi_graph(self):
        G = nx.barabasi_albert_graph(n=self.n_nodes, m=2, seed=self.seed)
        G = nx.DiGraph([(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()])
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
    
    def generate_traffic_data(self):
        traffic_patterns = {i: np.random.choice([1.0, 1.5, 2.0], size=self.n_time_periods, p=[0.5, 0.3, 0.2]) for i in range(self.n_nodes)}
        return traffic_patterns

    def generate_machine_statuses(self):
        machine_status = np.random.choice([0, 1], size=(self.n_nodes, self.n_time_periods), p=[self.machine_breakdown_prob, 1 - self.machine_breakdown_prob])
        maintenance_schedule = np.random.choice([0, 1], size=(self.n_nodes, self.n_time_periods), p=[self.maintenance_prob, 1 - self.maintenance_prob])
        return machine_status, maintenance_schedule

    def generate_staffing_levels(self):
        staffing_levels = np.random.randint(self.min_staff, self.max_staff, size=self.n_time_periods)
        return staffing_levels

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        if random.choice([True, False]):
            G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        else:
            G, adj_mat, edge_list, incommings, outcommings = self.generate_barabasi_graph()
        commodities = self.generate_commodities(G)
        traffic_patterns = self.generate_traffic_data()
        machine_status, maintenance_schedule = self.generate_machine_statuses()
        staffing_levels = self.generate_staffing_levels()
        
        res = {
            'commodities': commodities,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'traffic_patterns': traffic_patterns,
            'machine_status': machine_status,
            'maintenance_schedule': maintenance_schedule,
            'staffing_levels': staffing_levels
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        traffic_patterns = instance['traffic_patterns']
        machine_status = instance['machine_status']
        maintenance_schedule = instance['maintenance_schedule']
        staffing_levels = instance['staffing_levels']

        model = Model("FCMCNF")

        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", lb=0, name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        z_vars = {f"z_{k+1}": model.addVar(vtype="I", lb=0, name=f"z_{k+1}") for k in range(self.n_commodities)}
        m_vars = {f"m_{i+1}_{t+1}": model.addVar(vtype="B", name=f"m_{i+1}_{t+1}") for i in range(self.n_nodes) for t in range(self.n_time_periods)}
        s_vars = {f"s_{i+1}_{j+1}": model.addVar(vtype="I", lb=0, name=f"s_{i+1}_{j+1}") for (i, j) in edge_list}

        # Objective Function: Include penalties for unmet demand, expected traffic delays, energy consumption, and staffing levels
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            z_vars[f"z_{k+1}"] * 100  # Penalty for unmet demand
            for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][2] * traffic_patterns[i][j % self.n_time_periods] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            self.energy_consumption_rate * m_vars[f"m_{i+1}_{t+1}"]
            for i in range(self.n_nodes) for t in range(self.n_time_periods)
        )
        objective_expr += quicksum(
            self.staffing_cost * staffing_levels[t] for t in range(self.n_time_periods)
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
        
        # Machine Operational Constraints
        for i in range(self.n_nodes):
            for t in range(self.n_time_periods):
                if machine_status[i, t] == 0 or maintenance_schedule[i, t] == 1:
                    model.addCons(m_vars[f"m_{i+1}_{t+1}"] == 0, f"machine_{i+1}_{t+1}")
                else:
                    model.addCons(m_vars[f"m_{i+1}_{t+1}"] == 1, f"machine_{i+1}_{t+1}")

        # Unmet Demand Constraints
        for k in range(self.n_commodities):
            demand_expr = quicksum(x_vars[f"x_{commodities[k][0]+1}_{j+1}_{k+1}"] for j in outcommings[commodities[k][0]]) - quicksum(x_vars[f"x_{j+1}_{commodities[k][0]+1}_{k+1}"] for j in incommings[commodities[k][0]])
            model.addCons(demand_expr + z_vars[f"z_{k+1}"] >= commodities[k][2], f"demand_{k+1}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 10,
        'max_n_nodes': 22,
        'min_n_commodities': 225,
        'max_n_commodities': 472,
        'c_range': (220, 1000),
        'd_range': (135, 1350),
        'ratio': 3000,
        'k_max': 1120,
        'er_prob': 0.31,
        'n_time_periods': 75,
        'machine_breakdown_prob': 0.38,
        'maintenance_prob': 0.45,
        'min_staff': 35,
        'max_staff': 140,
        'staffing_cost': 150,
        'energy_consumption_rate': 2250,
        'storage_capacity': 500,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")