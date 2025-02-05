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

    # Data Generation
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
        
        # Additional data
        self.batch_sizes = [random.randint(1, 10) for _ in range(len(edge_list))]
        self.maintenance_schedule = [random.choice([0, 1]) for _ in range(self.n_nodes)]

        # Facility-related data generation
        self.n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        self.operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=self.n_facilities).tolist()
        self.assignment_cost = np.random.normal(loc=5, scale=2, size=len(edge_list)).tolist()
        self.capacity = np.random.randint(10, 50, size=self.n_facilities).tolist()

        res['batch_sizes'] = self.batch_sizes
        res['maintenance_schedule'] = self.maintenance_schedule
        res['n_facilities'] = self.n_facilities
        res['operating_cost'] = self.operating_cost
        res['assignment_cost'] = self.assignment_cost
        res['capacity'] = self.capacity
        
        return res

    # PySCIPOpt Modeling
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        batch_sizes = instance['batch_sizes']
        maintenance_schedule = instance['maintenance_schedule']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']
        
        model = Model("FCMCNF")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        b_vars = {f"b_{i+1}_{j+1}": model.addVar(vtype="I", name=f"b_{i+1}_{j+1}") for (i, j) in edge_list}
        m_vars = {f"m_{i+1}": model.addVar(vtype="B", name=f"m_{i+1}") for i in range(self.n_nodes)}
        q_vars = {f"q_{i+1}_{j+1}": model.addVar(vtype="C", name=f"q_{i+1}_{j+1}") for (i, j) in edge_list}
        z_vars = {(i+1, j+1): model.addVar(vtype="B", name=f"z_{i+1}_{j+1}") for (i, j) in edge_list for f in range(n_facilities)}
        w_vars = {f"w_{f+1}": model.addVar(vtype="C", name=f"w_{f+1}") for f in range(n_facilities)}

        # Objective function
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            batch_sizes[idx] * q_vars[f"q_{i+1}_{j+1}"]
            for idx, (i, j) in enumerate(edge_list)
        )
        objective_expr += quicksum(
            operating_cost[f] * w_vars[f"w_{f+1}"]
            for f in range(n_facilities)
        )
        objective_expr += quicksum(
            assignment_cost[idx] * z_vars[(i+1, j+1)]
            for idx, (i, j) in enumerate(edge_list)
        )

        model.setObjective(objective_expr, "minimize")

        # Constraints
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        # New constraints
        for (i, j) in edge_list:
            model.addCons(b_vars[f"b_{i+1}_{j+1}"] <= len(batch_sizes), f"batch_constraint_{i+1}_{j+1}")
            model.addCons(q_vars[f"q_{i+1}_{j+1}"] <= 5.0, f"queue_time_constraint_{i+1}_{j+1}")

        for i in range(self.n_nodes):
            model.addCons(m_vars[f"m_{i+1}"] == maintenance_schedule[i], f"maintenance_constraint_{i+1}")
            model.addCons(quicksum(y_vars[f"y_{i+1}_{j+1}"] for j in outcommings[i]) <= 10, f"production_run_constraint_{i+1}")

        for (i, j) in edge_list:
            for f in range(n_facilities):
                model.addCons(z_vars[(i+1, j+1)] <= y_vars[f"y_{i+1}_{j+1}"], f"edge_assignment_{i+1}_{j+1}_{f+1}")

        for f in range(n_facilities):
            model.addCons(quicksum(z_vars[(i+1, j+1)] for (i, j) in edge_list) <= capacity[f], f"capacity_constraint_{f+1}")

        for f in range(n_facilities):
            model.addCons(w_vars[f"w_{f+1}"] == quicksum(z_vars[(i+1, j+1)] for (i, j) in edge_list), f"throughput_constraint_{f+1}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 2,
        'max_n_nodes': 150,
        'min_n_commodities': 1,
        'max_n_commodities': 5,
        'c_range': (80, 370),
        'd_range': (0, 6),
        'ratio': 10,
        'k_max': 2,
        'er_prob': 0.17,
        'facility_min_count': 2,
        'facility_max_count': 225,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")