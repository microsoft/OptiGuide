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
        indices = np.random.choice(self.n_cols, size=nnzrs)  
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  

        _, col_nrows = np.unique(indices, return_counts=True)
        indices[:self.n_rows] = np.random.permutation(self.n_rows) 
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

        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        graph = nx.barabasi_albert_graph(self.n_cols, self.num_edges_per_node, seed=self.seed)
        capacities = np.random.randint(1, self.max_capacity, size=len(graph.edges))
        flows = np.random.uniform(0, self.max_flow, size=len(graph.edges))

        source_node, sink_node = 0, self.n_cols - 1

        adj_list = {i: [] for i in range(self.n_cols)}
        for idx, (u, v) in enumerate(graph.edges):
            adj_list[u].append((v, flows[idx], capacities[idx]))
            adj_list[v].append((u, flows[idx], capacities[idx]))  

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

        parking_capacity = np.random.randint(1, self.max_parking_capacity, size=self.n_parking_zones)
        parking_zones = {i: np.random.choice(range(self.n_cols), size=self.n_parking_in_zone, replace=False) for i in
                         range(self.n_parking_zones)}

        time_windows = {i: (np.random.randint(0, self.latest_delivery_time // 2), np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) for i in range(self.n_cols)}
        travel_times = {(u, v): np.random.randint(1, self.max_travel_time) for u, v in graph.edges}
        uncertainty = {i: np.random.normal(0, self.time_uncertainty_stddev, size=2) for i in range(self.n_cols)}

        res.update({'parking_capacity': parking_capacity, 
                    'parking_zones': parking_zones, 
                    'time_windows': time_windows, 
                    'travel_times': travel_times, 
                    'uncertainty': uncertainty})
        
        autonomous_vehicle_routes = {i: np.random.choice(range(self.n_cols), size=self.av_route_length, replace=False) for i in range(self.n_autonomous_vehicles)}
        road_closures = np.random.choice(range(self.n_cols), size=self.n_road_closures, replace=False)
        special_events = np.random.choice(range(self.n_cols), size=self.n_special_events, replace=False)
        
        res.update({'autonomous_vehicle_routes': autonomous_vehicle_routes,
                    'road_closures': road_closures,
                    'special_events': special_events})

        adaptive_signals = {i: np.random.choice([True, False]) for i in range(self.n_cols)}

        res.update({'adaptive_signals': adaptive_signals})

        eco_friendly_zones = np.random.choice(range(self.n_cols), size=self.n_eco_friendly_zones, replace=False)
        co2_saving = {i: np.random.uniform(0, self.max_co2_saving) for i in eco_friendly_zones}

        res.update({'eco_friendly_zones': eco_friendly_zones,
                    'co2_saving': co2_saving})

        res.update({'sustainability_constraint': np.random.uniform(0, self.min_sustainability_requirement)})

        # Add new multi-scenario data for robustness
        num_scenarios = 3
        uncertain_time_windows = {i: [(np.random.randint(0, self.latest_delivery_time // 2), np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) for _ in range(num_scenarios)] for i in range(self.n_cols)}
        res.update({'num_scenarios': num_scenarios, 'uncertain_time_windows': uncertain_time_windows})

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
        uncertainty = instance['uncertainty']
        autonomous_vehicle_routes = instance['autonomous_vehicle_routes']
        road_closures = instance['road_closures']
        special_events = instance['special_events']
        adaptive_signals = instance['adaptive_signals']
        eco_friendly_zones = instance['eco_friendly_zones']
        co2_saving = instance['co2_saving']
        sustainability_constraint = instance['sustainability_constraint']
        num_scenarios = instance['num_scenarios']
        uncertain_time_windows = instance['uncertain_time_windows']

        model = Model("EnhancedLogisticsOptimization")
        var_names = {}

        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        flow_vars = {}
        for u in adj_list:
            for v, _, capacity in adj_list[u]:
                flow_vars[(u, v)] = model.addVar(vtype='C', lb=0, ub=capacity, name=f"f_{u}_{v}")

        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        for node in adj_list:
            if node == source_node:
                model.addCons(quicksum(flow_vars[(source_node, v)] for v, _, _ in adj_list[source_node]) >= 1,
                              f"flow_source_{source_node}")
            elif node == sink_node:
                model.addCons(quicksum(flow_vars[(u, sink_node)] for u, _, _ in adj_list[sink_node]) >= 1,
                              f"flow_sink_{sink_node}")
            else:
                inflow = quicksum(flow_vars[(u, node)] for u, _, _ in adj_list[node] if (u, node) in flow_vars)
                outflow = quicksum(flow_vars[(node, v)] for v, _, _ in adj_list[node] if (node, v) in flow_vars)
                model.addCons(inflow - outflow == 0, f"flow_balance_{node}")

        parking_vars = {}
        for zone, cols in parking_zones.items():
            for col in cols:
                parking_vars[col] = model.addVar(vtype="B", name=f"p_{col}")
                model.addCons(var_names[col] <= parking_vars[col], f"occupy_{col}")
            model.addCons(quicksum(parking_vars[col] for col in cols) <= parking_capacity[zone], f"parking_limit_{zone}")

        time_vars = {}
        early_penalty_vars = {}
        late_penalty_vars = {}
        for j in range(self.n_cols):
            time_vars[j] = model.addVar(vtype='C', name=f"t_{j}")
            early_penalty_vars[j] = model.addVar(vtype='C', name=f"e_{j}")
            late_penalty_vars[j] = model.addVar(vtype='C', name=f"l_{j}")
            
            start_window, end_window = time_windows[j]
            uncertainty_start, uncertainty_end = uncertainty[j]
            model.addCons(time_vars[j] >= start_window + uncertainty_start, f"time_window_start_{j}")
            model.addCons(time_vars[j] <= end_window + uncertainty_end, f"time_window_end_{j}")
            
            model.addCons(early_penalty_vars[j] >= start_window + uncertainty_start - time_vars[j], f"early_penalty_{j}")
            model.addCons(late_penalty_vars[j] >= time_vars[j] - (end_window + uncertainty_end), f"late_penalty_{j}")

        cost_term = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        flow_term = quicksum(flow_vars[(u, v)] for u, v in flow_vars)
        parking_penalty_term = quicksum(parking_vars[col] for col in parking_vars)
        time_penalty_term = quicksum(early_penalty_vars[j] + late_penalty_vars[j] for j in range(self.n_cols))
        av_penalty_term = quicksum(quicksum(var_names[node] for node in route if node in road_closures or node in special_events) for route in autonomous_vehicle_routes.values())
        peak_period_penalty = self.peak_period_penalty_weight * quicksum(var_names[j] for j in self.peak_hours_nodes)

        for route in autonomous_vehicle_routes.values():
            for node in route:
                for closure in road_closures:
                    model.addCons(var_names[node] + var_names[closure] <= 1, f"no_intersect_{node}_{closure}")
                for event in special_events:
                    model.addCons(var_names[node] + var_names[event] <= 1, f"no_intersect_{node}_{event}")

        model.addCons(quicksum(co2_saving[node] * var_names[node] for node in eco_friendly_zones) >= sustainability_constraint, "sustainability")

        # New complex constraint: Robust constraints across scenarios
        for j in range(self.n_cols):
            for s in range(num_scenarios):
                start_window, end_window = uncertain_time_windows[j][s]
                model.addCons(time_vars[j] >= start_window, f"scenario_time_window_start_{j}_{s}")
                model.addCons(time_vars[j] <= end_window, f"scenario_time_window_end_{j}_{s}")

        objective_expr = cost_term - self.flow_weight * flow_term + self.parking_penalty_weight * parking_penalty_term + self.time_penalty_weight * time_penalty_term + av_penalty_term + peak_period_penalty

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 1500,
        'n_cols': 281,
        'density': 0.1,
        'max_coef': 1125,
        'num_edges_per_node': 5,
        'max_capacity': 2400,
        'max_flow': 50,
        'flow_weight': 0.59,
        'n_parking_zones': 20,
        'n_parking_in_zone': 100,
        'max_parking_capacity': 1000,
        'parking_penalty_weight': 0.66,
        'latest_delivery_time': 960,
        'max_travel_time': 60,
        'time_penalty_weight': 0.31,
        'time_uncertainty_stddev': 10,
        'n_autonomous_vehicles': 10,
        'av_route_length': 15,
        'n_road_closures': 5,
        'n_special_events': 2,
        'peak_hours_nodes': list(range(50, 100)),
        'peak_period_penalty_weight': 1.2,
        'n_eco_friendly_zones': 50,
        'max_co2_saving': 100,
        'min_sustainability_requirement': 500,
    }

    enhanced_logistics_optimization = EnhancedLogisticsOptimization(parameters, seed=seed)
    instance = enhanced_logistics_optimization.generate_instance()
    solve_status, solve_time = enhanced_logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")