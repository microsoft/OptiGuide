import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexKnapsackGISP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        weights = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(self.min_range, self.max_range, self.number_of_items)
        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (self.max_range-self.min_range), 1),
                               weights + (self.max_range-self.min_range)]))
        elif self.scheme == 'strongly correlated':
            profits = weights + (self.max_range - self.min_range) / 10
        elif self.scheme == 'subset-sum':
            profits = weights
        else:
            raise NotImplementedError

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        res = {'weights': weights, 'profits': profits, 'capacities': capacities}

        # Add graph, edge costs, and time windows from GISP
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        res.update({'G': G, 'E2': E2})

        parking_capacity = np.random.randint(1, self.max_parking_capacity, size=self.n_parking_zones)
        parking_zones = {i: np.random.choice(range(len(G.nodes)), size=self.n_parking_in_zone, replace=False) for i in range(self.n_parking_zones)}
        res.update({'parking_capacity': parking_capacity, 'parking_zones': parking_zones})

        time_windows = {node: (np.random.randint(0, self.latest_delivery_time // 2), 
                               np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) for node in G.nodes}
        uncertainty = {node: np.random.normal(0, self.time_uncertainty_stddev, size=2) for node in G.nodes}
        res.update({'time_windows': time_windows, 'uncertainty': uncertainty})

        return res

    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        if self.set_type == 'SET1':
            for node in G.nodes:
                G.nodes[node]['revenue'] = np.random.randint(1, 100)
            for u, v in G.edges:
                G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.set_param)
        elif self.set_type == 'SET2':
            for node in G.nodes:
                G.nodes[node]['revenue'] = float(self.set_param)
            for u, v in G.edges:
                G[u][v]['cost'] = 1.0

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        G, E2 = instance['G'], instance['E2']
        parking_capacity, parking_zones = instance['parking_capacity'], instance['parking_zones']
        time_windows, uncertainty = instance['time_windows'], instance['uncertainty']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("ComplexKnapsackGISP")
        var_names = {}
        node_vars, edge_vars, parking_vars, time_vars = {}, {}, {}, {}
        early_penalty_vars, late_penalty_vars = {}, {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Node and edge variables for GISP
        for node in G.nodes:
            node_vars[f"x{node}"] = model.addVar(vtype="B", name=f"x{node}")
        for u, v in G.edges:
            edge_vars[f"y{u}_{v}"] = model.addVar(vtype="B", name=f"y{u}_{v}")

        # Parking and time variables
        for node in G.nodes:
            parking_vars[f"p{node}"] = model.addVar(vtype="B", name=f"p{node}")
            time_vars[f"t{node}"] = model.addVar(vtype="C", name=f"t{node}")
            early_penalty_vars[f"e{node}"] = model.addVar(vtype="C", name=f"e{node}")
            late_penalty_vars[f"l{node}"] = model.addVar(vtype="C", name=f"l{node}")

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        objective_expr += quicksum(G.nodes[node]['revenue'] * node_vars[f"x{node}"] for node in G.nodes)
        objective_expr -= quicksum(G[u][v]['cost'] * edge_vars[f"y{u}_{v}"] for u, v in E2)
        objective_expr -= self.time_penalty_weight * quicksum(early_penalty_vars[f"e{node}"] + late_penalty_vars[f"l{node}"] for node in G.nodes)

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= 1,
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )

        # Constraint: Ensure no overlap in assignments for items represented by graph nodes
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

        # Parking constraints
        for zone, nodes in parking_zones.items():
            model.addCons(
                quicksum(parking_vars[f"p{node}"] for node in nodes if f"p{node}" in parking_vars) <= parking_capacity[zone], 
                f"parking_limit_{zone}"
            )
            for node in nodes:
                if f"p{node}" in parking_vars:
                    model.addCons(
                        parking_vars[f"p{node}"] <= node_vars[f"x{node}"], 
                        f"assign_parking_{node}"
                    )
        # Ensure time windows and penalties for early/late deliveries
        for node in G.nodes:
            if f"t{node}" in time_vars:
                start_window, end_window = time_windows[node]
                uncertainty_start, uncertainty_end = uncertainty[node]
                model.addCons(time_vars[f"t{node}"] >= start_window + uncertainty_start, 
                              f"time_window_start_{node}")
                model.addCons(time_vars[f"t{node}"] <= end_window + uncertainty_end, 
                              f"time_window_end_{node}")
                
                model.addCons(early_penalty_vars[f"e{node}"] >= start_window + uncertainty_start - time_vars[f"t{node}"], 
                              f"early_penalty_{node}")
                model.addCons(late_penalty_vars[f"l{node}"] >= time_vars[f"t{node}"] - (end_window + uncertainty_end), 
                              f"late_penalty_{node}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 200,
        'number_of_knapsacks': 10,
        'min_range': 10,
        'max_range': 30,
        'scheme': 'weakly correlated',
        'min_n': 50,
        'max_n': 100,
        'er_prob': 0.3,
        'set_type': 'SET1',
        'set_param': 75.0,
        'alpha': 0.2,
        'max_parking_capacity': 200,
        'n_parking_zones': 5,
        'n_parking_in_zone': 10,
        'latest_delivery_time': 720,
        'time_uncertainty_stddev': 7,
        'time_penalty_weight': 0.1,
    }

    complex_knapsack = ComplexKnapsackGISP(parameters, seed=seed)
    instance = complex_knapsack.generate_instance()
    solve_status, solve_time = complex_knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")