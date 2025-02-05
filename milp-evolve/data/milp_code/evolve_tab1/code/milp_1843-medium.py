import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GISP_NDP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.set_param)

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_traffic_demand(self, G):
        for node in G.nodes:
            G.nodes[node]['traffic_demand'] = np.random.randint(1, 100)

    def generate_dynamic_costs(self, G):
        for u, v in G.edges:
            G[u][v]['dynamic_cost'] = np.random.normal(G[u][v]['cost'], G[u][v]['cost'] * self.cost_deviation)
            G[u][v]['network_strength'] = np.random.uniform(0.5, 1.5)

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        self.generate_traffic_demand(G)
        self.generate_dynamic_costs(G)
        
        energy_limits = {node: np.random.randint(50, 150) for node in G.nodes}
        shift_demands = {shift: np.random.randint(500, 1000) for shift in range(self.n_shifts)}
        
        scenarios = []
        for _ in range(self.n_scenarios):
            scenario = {}
            for node in G.nodes:
                scenario[node] = {
                    'traffic_demand': np.random.normal(G.nodes[node]['traffic_demand'], G.nodes[node]['traffic_demand'] * self.traffic_deviation)
                }
            for u, v in G.edges:
                scenario[(u, v)] = {
                    'dynamic_cost': np.random.normal(G[u][v]['dynamic_cost'], G[u][v]['dynamic_cost'] * self.cost_deviation)
                }
            scenarios.append(scenario)
        
        # New data for time windows and driver availability
        time_windows = {node: (np.random.randint(0, 4), np.random.randint(5, 9)) for node in G.nodes}
        driver_availability = {node: np.random.randint(1, 10) for node in G.nodes}
        grid_dependency_costs = {node: np.random.randint(5, 50) for node in G.nodes}
        
        return {'G': G, 'E2': E2, 'scenarios': scenarios, 'energy_limits': energy_limits, 'shift_demands': shift_demands,
                'time_windows': time_windows, 'driver_availability': driver_availability, 'grid_dependency_costs': grid_dependency_costs}
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2, scenarios, energy_limits, shift_demands, time_windows, driver_availability, grid_dependency_costs = \
            instance['G'], instance['E2'], instance['scenarios'], instance['energy_limits'], instance['shift_demands'], \
            instance['time_windows'], instance['driver_availability'], instance['grid_dependency_costs']
        
        model = Model("GISP_NDP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}

        traffic_vars = {s: {f"t{node}_s{s}": model.addVar(vtype="C", name=f"t{node}_s{s}") for node in G.nodes} for s in range(self.n_scenarios)}
        dynamic_cost_vars = {s: {f"dc{u}_{v}_s{s}": model.addVar(vtype="C", name=f"dc{u}_{v}_s{s}") for u, v in G.edges} for s in range(self.n_scenarios)}

        # New vars for energy consumption status, shift loads, and additional constraints
        energy_vars = {node: model.addVar(vtype="C", name=f"e{node}") for node in G.nodes}
        shift_vars = {shift: model.addVar(vtype="C", name=f"s{shift}") for shift in range(self.n_shifts)}
        
        time_window_vars = {node: model.addVar(vtype="C", name=f"tw{node}") for node in G.nodes}
        driver_availability_vars = {node: model.addVar(vtype="C", name=f"da{node}") for node in G.nodes}
        grid_dependency_vars = {node: model.addVar(vtype="C", name=f"gd{node}") for node in G.nodes}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        ) - quicksum(
            scenarios[s][(u, v)]['dynamic_cost'] * edge_vars[f"y{u}_{v}"]
            for s in range(self.n_scenarios) for u, v in E2
        )

        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )

        for s in range(self.n_scenarios):
            for node in G.nodes:
                model.addCons(
                    traffic_vars[s][f"t{node}_s{s}"] == scenarios[s][node]['traffic_demand'],
                    name=f"RobustTraffic_{node}_s{s}"
                )
            for u, v in G.edges:
                model.addCons(
                    dynamic_cost_vars[s][f"dc{u}_{v}_s{s}"] == scenarios[s][(u, v)]['dynamic_cost'],
                    name=f"RobustDynamicCost_{u}_{v}_s{s}"
                )
        
        # Energy consumption constraints
        for node in G.nodes:
            model.addCons(
                energy_vars[node] <= energy_limits[node],
                name=f"EnergyLimit_{node}"
            )

        # Shift balance constraints
        for shift in range(self.n_shifts):
            model.addCons(
                shift_vars[shift] >= shift_demands[shift] / self.n_shifts,
                name=f"ShiftDemand_{shift}"
            )
            model.addCons(
                shift_vars[shift] <= shift_demands[shift] * 1.5 / self.n_shifts,
                name=f"ShiftBalance_{shift}"
            )

        # Time window constraints
        for node in G.nodes:
            model.addCons(
                time_window_vars[node] >= time_windows[node][0],
                name=f"TimeWindowStart_{node}"
            )
            model.addCons(
                time_window_vars[node] <= time_windows[node][1],
                name=f"TimeWindowEnd_{node}"
            )

        # Driver availability constraints
        for node in G.nodes:
            model.addCons(
                driver_availability_vars[node] <= driver_availability[node],
                name=f"DriverAvailability_{node}"
            )

        # Grid dependency costs
        for node in G.nodes:
            model.addCons(
                grid_dependency_vars[node] == grid_dependency_costs[node],
                name=f"GridDependency_{node}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 75,
        'max_n': 390,
        'er_prob': 0.1,
        'set_type': 'SET1',
        'set_param': 700.0,
        'alpha': 0.38,
        'n_scenarios': 9,
        'traffic_deviation': 0.31,
        'cost_deviation': 0.1,
        'n_shifts': 2,
    }

    gisp_ndp = GISP_NDP(parameters, seed=seed)
    instance = gisp_ndp.generate_instance()
    solve_status, solve_time = gisp_ndp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")