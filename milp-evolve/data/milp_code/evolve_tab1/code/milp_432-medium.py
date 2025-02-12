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
            if i >= self.n_rows:
                indices[i:i + n] = np.random.choice(self.n_rows, size=n, replace=False)
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i + n] = np.random.choice(remaining_rows, size=i + n - self.n_rows, replace=False)
            i += n
            indptr.append(i)

        # objective coefficients
        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        # generate a random graph to introduce additional complexity
        graph = nx.erdos_renyi_graph(self.n_cols, self.edge_prob, seed=self.seed)
        capacities = np.random.randint(1, self.max_capacity, size=len(graph.edges))
        flows = np.random.uniform(0, self.max_flow, size=len(graph.edges))
        fuel_consumption = np.random.uniform(1, self.max_fuel_consumption, size=len(graph.edges))
        emissions = np.random.uniform(1, self.max_emissions, size=len(graph.edges))

        # generate start and end nodes for flow network
        source_node, sink_node = 0, self.n_cols - 1

        # convert graph to adjacency list
        adj_list = {i: [] for i in range(self.n_cols)}
        for idx, (u, v) in enumerate(graph.edges):
            adj_list[u].append((v, flows[idx], capacities[idx], fuel_consumption[idx], emissions[idx]))
            adj_list[v].append((u, flows[idx], capacities[idx], fuel_consumption[idx], emissions[idx]))  # for undirected edges

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

        # Generate delivery time windows
        time_windows = {i: (np.random.randint(0, self.latest_delivery_time // 2), np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) for i in range(self.n_cols)}

        # Generate stochastic demand
        demand = np.random.normal(self.mean_demand, self.std_dev_demand, size=self.n_rows)

        # Generate uncertain travel times
        travel_time_uncertainty = {edge: np.random.normal(self.mean_travel_time, self.std_dev_travel_time)
                                   for edge in graph.edges}

        res.update({'parking_capacity': parking_capacity, 
                    'parking_zones': parking_zones,
                    'time_windows': time_windows,
                    'demand': demand,
                    'travel_time_uncertainty': travel_time_uncertainty})

        # Additional data for Set Packing and Set Covering constraints
        delivery_task_types = np.random.randint(1, self.max_task_types + 1, size=self.n_types)
        critical_nodes = np.random.choice(range(self.n_cols), size=self.n_critical_nodes, replace=False)
        res.update({'delivery_task_types': delivery_task_types, 'critical_nodes': critical_nodes})

        return res

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
        demand = instance['demand']
        travel_time_uncertainty = instance['travel_time_uncertainty']
        delivery_task_types = instance['delivery_task_types']
        critical_nodes = instance['critical_nodes']

        model = Model("EnhancedLogisticsOptimization")
        var_names = {}

        # Create binary variables and set objective coefficients
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        flow_vars = {}
        fuel_vars = {}
        emission_vars = {}
        travel_time_robust_vars = {}

        # Create flow, fuel, and emission variables for edges
        for u in adj_list:
            for v, _, capacity, fuel, emissions in adj_list[u]:
                flow_vars[(u, v)] = model.addVar(vtype='C', lb=0, ub=capacity, name=f"f_{u}_{v}")
                fuel_vars[(u, v)] = model.addVar(vtype='C', lb=0, name=f"fu_{u}_{v}", obj=fuel)
                emission_vars[(u, v)] = model.addVar(vtype='C', lb=0, name=f"em_{u}_{v}", obj=emissions)
                travel_time_robust_vars[(u, v)] = model.addVar(vtype='C', lb=0, name=f"tt_{u}_{v}")

        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= demand[row], f"c_{row}")

        # Flow balance constraints
        for node in adj_list:
            inflow = quicksum(flow_vars[(u, node)] for u, _, _, _, _ in adj_list[node] if (u, node) in flow_vars)
            outflow = quicksum(flow_vars[(node, v)] for v, _, _, _, _ in adj_list[node] if (node, v) in flow_vars)
            if node == source_node:
                model.addCons(outflow == quicksum(var_names[j] for j in range(self.n_cols)), f"flow_out_{node}")
            elif node == sink_node:
                model.addCons(inflow == quicksum(var_names[j] for j in range(self.n_cols)), f"flow_in_{node}")
            else:
                model.addCons(inflow - outflow == 0, f"flow_balance_{node}")

        # Create and constrain fuel and emission variables
        total_fuel_consumption = quicksum(fuel_vars[e] for e in fuel_vars)
        total_emissions = quicksum(emission_vars[e] for e in emission_vars)
        model.addCons(total_fuel_consumption <= self.max_total_fuel_consumption, "total_fuel_limit")
        model.addCons(total_emissions <= self.max_total_emission_limit, "total_emission_limit")

        # Travel time robustness constraints
        for (u, v) in travel_time_uncertainty:
            model.addCons(travel_time_robust_vars[(u, v)] >= flow_vars[(u, v)] * travel_time_uncertainty[(u, v)])

        # Parking space constraints
        parking_vars = {}
        for zone, cols in parking_zones.items():
            for col in cols:
                parking_vars[col] = model.addVar(vtype="B", name=f"p_{col}")
                model.addCons(var_names[col] <= parking_vars[col], f"occupy_{col}")

            # Ensure the number of occupied parking spaces in a zone is limited
            model.addCons(quicksum(parking_vars[col] for col in cols) <= parking_capacity[zone], f"parking_limit_{zone}")

        # Delivery time window constraints
        time_vars = {}
        early_penalty_vars = {}
        late_penalty_vars = {}
        for j in range(self.n_cols):
            time_vars[j] = model.addVar(vtype='C', name=f"t_{j}")
            early_penalty_vars[j] = model.addVar(vtype='C', name=f"e_{j}")
            late_penalty_vars[j] = model.addVar(vtype='C', name=f"l_{j}")
            
            start_window, end_window = time_windows[j]
            model.addCons(time_vars[j] >= start_window, f"time_window_start_{j}")
            model.addCons(time_vars[j] <= end_window, f"time_window_end_{j}")
            
            model.addCons(early_penalty_vars[j] >= start_window - time_vars[j], f"early_penalty_{j}")
            model.addCons(late_penalty_vars[j] >= time_vars[j] - end_window, f"late_penalty_{j}")

        # Set Packing constraints
        delivery_task_vars = {}
        for t in range(self.n_types):
            delivery_task_vars[t] = model.addVar(vtype="B", name=f"dt_{t}")

        # Ensure that each column (delivery point) participates in at most one task type
        for j in range(self.n_cols):
            model.addCons(quicksum(delivery_task_vars[t] for t in delivery_task_types) <= 1, f"task_selection_{j}")

        # Set Covering constraints
        for node in critical_nodes:
            model.addCons(quicksum(var_names[node] for _ in critical_nodes) >= 1, f"cover_critical_node_{node}")

        # Objective function
        cost_term = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        parking_penalty_term = quicksum(parking_vars[col] for col in parking_vars)
        time_penalty_term = quicksum(early_penalty_vars[j] + late_penalty_vars[j] for j in range(self.n_cols))
        
        # Objective function modifications
        objective_expr = cost_term + self.parking_penalty_weight * parking_penalty_term + self.time_penalty_weight * time_penalty_term
        
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 375,
        'n_cols': 210,
        'density': 0.59,
        'max_coef': 1125,
        'edge_prob': 0.17,
        'max_capacity': 1200,
        'max_flow': 50,
        'max_fuel_consumption': 11,
        'max_emissions': 500,
        'flow_weight': 0.59,
        'n_parking_zones': 5,
        'n_parking_in_zone': 50,
        'max_parking_capacity': 1000,
        'parking_penalty_weight': 0.1,
        'latest_delivery_time': 96,
        'time_penalty_weight': 0.38,
        'max_total_fuel_consumption': 15,
        'max_total_emission_limit': 750,
        'mean_demand': 50,
        'std_dev_demand': 0,
        'mean_travel_time': 0,
        'std_dev_travel_time': 10,
        'n_types': 10,
        'max_task_types': 3,
        'n_critical_nodes': 20,
    }

    enhanced_logistics_optimization = EnhancedLogisticsOptimization(parameters, seed=seed)
    instance = enhanced_logistics_optimization.generate_instance()
    solve_status, solve_time = enhanced_logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")