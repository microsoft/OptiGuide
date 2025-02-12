import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedLogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_rows * self.n_cols * self.density)

        # compute number of rows per column
        indices = np.random.choice(self.n_cols, size=nnzrs)  # random column indexes
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  # force at least 2 rows per col
        _, col_nrows = np.unique(indices, return_counts=True)

        # for each column, sample random rows
        indices[:self.n_rows] = np.random.permutation(self.n_rows) # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # empty column, fill with random rows
            if i >= self.n_rows:
                indices[i:i + n] = np.random.choice(self.n_rows, size=n, replace=False)

            # partially filled column, complete with random rows among remaining ones
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i + n] = np.random.choice(remaining_rows, size=i + n - self.n_rows,
                                                              replace=False)

            i += n
            indptr.append(i)

        # objective coefficients
        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        # generate a random graph to introduce additional complexity
        graph = nx.barabasi_albert_graph(self.n_cols, self.num_edges_per_node, seed=self.seed)
        capacities = np.random.randint(1, self.max_capacity, size=len(graph.edges))
        flows = np.random.uniform(0, self.max_flow, size=len(graph.edges))

        # generate start and end nodes for flow network
        source_node, sink_node = 0, self.n_cols - 1

        # convert graph to adjacency list
        adj_list = {i: [] for i in range(self.n_cols)}
        for idx, (u, v) in enumerate(graph.edges):
            adj_list[u].append((v, flows[idx], capacities[idx]))
            adj_list[v].append((u, flows[idx], capacities[idx]))  # for undirected edges

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {'c': c,
               'indptr_csr': indptr_csr,
               'indices_csr': indices_csr,
               'adj_list': adj_list,
               'source_node': source_node,
               'sink_node': sink_node}

        # Generate parking data
        parking_capacity = np.random.randint(1, self.max_parking_capacity, size=self.n_parking_zones)
        parking_zones = {i: np.random.choice(range(self.n_cols), size=self.n_parking_in_zone, replace=False) for i in
                         range(self.n_parking_zones)}

        # Generate delivery time windows and traffic congestion data
        time_windows = {i: (np.random.randint(0, self.latest_delivery_time // 2), np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) for i in range(self.n_cols)}
        travel_times = {(u, v): np.random.randint(1, self.max_travel_time) for u, v in graph.edges}

        res.update({'parking_capacity': parking_capacity, 
                    'parking_zones': parking_zones, 
                    'time_windows': time_windows, 
                    'travel_times': travel_times})

        # Battery storage and wear-out rate
        battery_storage = {node: np.random.randint(50, 200) for node in range(self.n_cols)}
        wear_out_rate = {node: np.random.uniform(0.01, 0.05) for node in range(self.n_cols)}

        # Collision probabilities and noise levels
        collision_probabilities = {(u, v): np.random.uniform(0, 0.05) for u, v in graph.edges}
        noise_levels = {(u, v): np.random.randint(1, 5) for u, v in graph.edges}
        
        # Workload distribution
        workload_distribution = {node: np.random.randint(1, 100) for node in range(self.n_cols)}

        res.update({'battery_storage': battery_storage,
                    'wear_out_rate': wear_out_rate,
                    'collision_probabilities': collision_probabilities,
                    'noise_levels': noise_levels,
                    'workload_distribution': workload_distribution})

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        adj_list = instance['adj_list']
        source_node = instance['source_node']
        sink_node = instance['sink_node']
        parking_capacity = instance['parking_capacity']
        parking_zones = instance['parking_zones']
        time_windows = instance['time_windows']
        travel_times = instance['travel_times']

        battery_storage = instance['battery_storage']
        wear_out_rate = instance['wear_out_rate']
        collision_probabilities = instance['collision_probabilities']
        noise_levels = instance['noise_levels']
        workload_distribution = instance['workload_distribution']

        model = Model("EnhancedLogisticsOptimization")
        var_names = {}

        # Create binary variables and set objective coefficients
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        flow_vars = {}
        # Create flow variables for edges
        for u in adj_list:
            for v, _, capacity in adj_list[u]:
                flow_vars[(u, v)] = model.addVar(vtype='C', lb=0, ub=capacity, name=f"f_{u}_{v}")

        # Add constraints to ensure each row is covered
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        # Add flow constraints
        for node in adj_list:
            if node == source_node:
                # flow balance for source node:
                model.addCons(quicksum(flow_vars[(source_node, v)] for v, _, _ in adj_list[source_node]) == 0,
                              f"flow_source_{source_node}")
            elif node == sink_node:
                # flow balance for sink node:
                model.addCons(quicksum(flow_vars[(u, sink_node)] for u, _, _ in adj_list[sink_node]) == 0,
                              f"flow_sink_{sink_node}")
            else:
                inflow = quicksum(flow_vars[(u, node)] for u, _, _ in adj_list[node] if (u, node) in flow_vars)
                outflow = quicksum(flow_vars[(node, v)] for v, _, _ in adj_list[node] if (node, v) in flow_vars)
                model.addCons(inflow - outflow == 0, f"flow_balance_{node}")

        # New Constraints: Parking space constraints
        parking_vars = {}
        for zone, cols in parking_zones.items():
            for col in cols:
                parking_vars[col] = model.addVar(vtype="B", name=f"p_{col}")
                model.addCons(var_names[col] <= parking_vars[col], f"occupy_{col}")

            # Constraint to ensure the number of occupied parking spaces in a zone is limited
            model.addCons(quicksum(parking_vars[col] for col in cols) <= parking_capacity[zone], f"parking_limit_{zone}")

        # Add variables and constraints for delivery time windows
        time_vars = {}
        early_penalty_vars = {}
        late_penalty_vars = {}
        for j in range(self.n_cols):
            time_vars[j] = model.addVar(vtype='C', name=f"t_{j}")
            early_penalty_vars[j] = model.addVar(vtype='C', name=f"e_{j}")
            late_penalty_vars[j] = model.addVar(vtype='C', name=f"l_{j}")
            model.addCons(time_vars[j] >= time_windows[j][0], f"time_window_start_{j}")
            model.addCons(time_vars[j] <= time_windows[j][1], f"time_window_end_{j}")
            model.addCons(early_penalty_vars[j] >= time_windows[j][0] - time_vars[j], f"early_penalty_{j}")
            model.addCons(late_penalty_vars[j] >= time_vars[j] - time_windows[j][1], f"late_penalty_{j}")

        # Battery Related Constraints
        battery_usage_vars = {}
        for node in range(self.n_cols):
            battery_usage_vars[node] = model.addVar(vtype="C", name=f"bu_{node}")
            model.addCons(battery_usage_vars[node] <= battery_storage[node], name=f"Battery_Storage_{node}")
            model.addCons(battery_usage_vars[node] <= var_names[node] * (1 - wear_out_rate[node]) * battery_storage[node], name=f"Battery_Wear_{node}")

        # Collision Minimization Constraints
        collision_vars = {}
        for (u, v) in collision_probabilities:
            collision_vars[(u, v)] = model.addVar(vtype="B", name=f"coll_{u}_{v}")
            model.addCons(collision_vars[(u, v)] <= flow_vars[(u, v)], name=f"Collision_{u}_{v}")

        # Noise Minimization Constraints
        noise_vars = {}
        for (u, v) in noise_levels:
            noise_vars[(u, v)] = model.addVar(vtype="C", name=f"noise_{u}_{v}")
            model.addCons(noise_vars[(u, v)] <= flow_vars[(u, v)] * noise_levels[(u, v)], name=f"Noise_{u}_{v}")

        # Workload Distribution Constraints
        workload_vars = {}
        max_workload = model.addVar(vtype="C", name="max_workload")
        for node in range(self.n_cols):
            workload_vars[node] = model.addVar(vtype="C", name=f"work_{node}")
            model.addCons(workload_vars[node] <= max_workload, name=f"Workload_{node}")
            model.addCons(workload_vars[node] == workload_distribution[node] * var_names[node], name=f"Workload_Equal_{node}")

        # Composite objective: Minimize total cost and maximize total flow
        cost_term = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        flow_term = quicksum(flow_vars[(u, v)] for u, v in flow_vars)
        parking_penalty_term = quicksum(parking_vars[col] for col in parking_vars)
        time_penalty_term = quicksum(early_penalty_vars[j] + late_penalty_vars[j] for j in range(self.n_cols))
        collision_term = quicksum(collision_probabilities[(u, v)] * collision_vars[(u, v)] for u, v in collision_vars)
        noise_term = quicksum(noise_levels[(u, v)] * noise_vars[(u, v)] for u, v in noise_vars)
        workload_term = quicksum(workload_vars[node] for node in workload_vars)

        # Objective function
        objective_expr = cost_term - self.flow_weight * flow_term + self.parking_penalty_weight * parking_penalty_term + self.time_penalty_weight * time_penalty_term + self.collision_penalty_weight * collision_term + self.noise_penalty_weight * noise_term + self.workload_penalty_weight * workload_term

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 1125,
        'n_cols': 281,
        'density': 0.17,
        'max_coef': 1125,
        'num_edges_per_node': 2,
        'max_capacity': 120,
        'max_flow': 500,
        'flow_weight': 0.31,
        'n_parking_zones': 2,
        'n_parking_in_zone': 25,
        'max_parking_capacity': 1000,
        'parking_penalty_weight': 0.73,
        'latest_delivery_time': 96,
        'max_travel_time': 45,
        'time_penalty_weight': 0.73,
        'collision_penalty_weight': 0.59,
        'noise_penalty_weight': 0.17,
        'workload_penalty_weight': 0.66,
    }

    enhanced_logistics_optimization = EnhancedLogisticsOptimization(parameters, seed=seed)
    instance = enhanced_logistics_optimization.generate_instance()
    solve_status, solve_time = enhanced_logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")