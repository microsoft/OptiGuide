import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict
from networkx.algorithms import bipartite

class RefrigeratedStorageLogistics:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_bipartite_graph(self, n1, n2, p):
        return bipartite.random_graph(n1, n2, p, seed=self.seed)

    def generate_maxsat_graph(self, n):
        divider = np.random.randint(1, 6)
        G = self.generate_bipartite_graph(n // divider, n - n // divider, self.er_prob)

        n_edges = len(G.edges)
        edges = list(G.edges)

        added_edges = 0
        while added_edges < n_edges * self.edge_addition_prob:
            i, j = np.random.randint(0, n), np.random.randint(0, n)
            if (i, j) not in edges and (j, i) not in edges:
                added_edges += 1
                edges.append((i, j))

        return edges

    def generate_instances(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_maxsat_graph(n)
        clauses = [(f'v{i},v{j}', 1) for i, j in edges] + [(f'-v{i},-v{j}', 1) for i, j in edges]

        # Added complexity: Data for seasonal variations, transportation cost, emissions, etc.
        seasonal_variation = np.random.normal(loc=1.0, scale=0.1, size=n)
        transportation_cost = np.random.random(size=(n, n))
        carbon_emissions = np.random.random(size=(n, n))
        demand = np.random.randint(10, 50, size=n)

        # Added complexity: New data
        temp_coeff = np.random.random(size=n)  # Temperature coefficients
        time_windows = [(random.randint(0, 12), random.randint(12, 24)) for _ in range(n)]  # Delivery time windows
        hybrid_storage_cap = np.random.randint(0, 2, size=n)  # Hybrid storage capacity
        priority_levels = np.random.randint(1, 5, size=(n, n))  # Priority levels (1 highest, 5 lowest)
        travel_time = np.random.random(size=(n, n))  # Travel times adapting with traffic

        res = {
            'clauses': clauses,
            'seasonal_variation': seasonal_variation,
            'transportation_cost': transportation_cost,
            'carbon_emissions': carbon_emissions,
            'demand': demand,
            'temp_coeff': temp_coeff,
            'time_windows': time_windows,
            'hybrid_storage_cap': hybrid_storage_cap,
            'priority_levels': priority_levels,
            'travel_time': travel_time
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        seasonal_variation = instance['seasonal_variation']
        transportation_cost = instance['transportation_cost']
        carbon_emissions = instance['carbon_emissions']
        demand = instance['demand']
        temp_coeff = instance['temp_coeff']
        time_windows = instance['time_windows']
        hybrid_storage_cap = instance['hybrid_storage_cap']
        priority_levels = instance['priority_levels']
        travel_time = instance['travel_time']

        model = Model("RefrigeratedStorageLogistics")
        var_names = {}  

        # Create variables for each literal and clause
        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var

        # Variables for constraints on transportation and emissions
        x = {}
        for i in range(len(seasonal_variation)):
            x[i] = model.addVar(vtype="B", name=f"x_{i}")

        # Objective function - maximize the number of satisfied clauses
        objective_expr = quicksum(
            var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        ) - quicksum(transportation_cost[i][j] * x[i] for i in range(len(x)) for j in range(len(x)))

        # Add constraints for each clause
        for idx, (clause, weight) in enumerate(clauses):
            vars_in_clause = clause.split(',')
            clause_var = var_names[f"cl_{idx}"]
            
            # Define the positive and negative parts
            positive_part = quicksum(var_names[var] for var in vars_in_clause if not var.startswith('-'))
            negative_part = quicksum(1 - var_names[var[1:]] for var in vars_in_clause if var.startswith('-'))
            
            # Total satisfied variables in the clause
            total_satisfied = positive_part + negative_part
            
            if weight < np.inf:
                model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")
            else:
                model.addCons(total_satisfied >= 1, name=f"clause_{idx}")

        # Add new constraints for seasonal variance, carbon emissions, demand distribution
        for i in range(len(seasonal_variation)):
            model.addCons(x[i] <= seasonal_variation[i], name=f"season_var_{i}")
        
        model.addCons(quicksum(carbon_emissions[i][j] * x[i] for i in range(len(x)) for j in range(len(x))) <= self.carbon_emission_limit, name="emission_limit")

        total_demand = quicksum(demand)
        for i in range(len(demand)):
            model.addCons(x[i] <= demand[i] / total_demand, name=f"equity_{i}")

        # New constraints for temperature coefficients
        for i in range(len(temp_coeff)):
            model.addCons(temp_coeff[i] * x[i] <= 1.0, name=f"temp_coeff_{i}")

        # New constraints for time-windows
        for i, (start, end) in enumerate(time_windows):
            t = model.addVar(vtype="I", lb=start, ub=end, name=f"time_{i}")
            model.addCons(x[i] * start <= t, name=f"timewin_start_{i}")
            model.addCons(x[i] * t <= end, name=f"timewin_end_{i}")

        # New constraints for hybrid storage capacity
        for i in range(len(hybrid_storage_cap)):
            model.addCons(x[i] <= hybrid_storage_cap[i], name=f"hybrid_storage_{i}")

        # New constraints for priority levels on transportation routes
        for i in range(len(priority_levels)):
            for j in range(len(priority_levels)):
                model.addCons(priority_levels[i][j] * x[i] <= 5, name=f"priority_{i}_{j}")

        # New constraints for adaptive traffic regulations
        for i in range(len(travel_time)):
            for j in range(len(travel_time)):
                model.addCons(travel_time[i][j] * x[i] <= 2, name=f"traffic_{i}_{j}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 37,
        'max_n': 187,
        'er_prob': 0.38,
        'edge_addition_prob': 0.24,
        'carbon_emission_limit': 375,
    }

    maxsat = RefrigeratedStorageLogistics(parameters, seed=seed)
    instance = maxsat.generate_instances()
    solve_status, solve_time = maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")