import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite
from itertools import combinations


class MaxSatisfiability:
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

        # Randomly generate seasons affecting road conditions
        seasons = ['dry', 'rainy']
        road_status = {edge: random.choice(seasons) for edge in edges}

        # Randomly generate future weather prediction affecting road conditions
        future_weather = {edge: random.choice(seasons) for edge in edges}
        future_prob = np.random.rand(len(edges))

        # Generate capacity and cost data for each edge
        capacities = {edge: np.random.randint(1, self.max_capacity + 1) for edge in edges}
        costs = {edge: np.random.randint(1, self.max_cost + 1) for edge in edges}

        # Generate clauses considering road conditions and demand patterns
        clauses = [
            (f'v{i},v{j}', 1) for i, j in edges if road_status[(i, j)] == 'dry'
        ] + [(f'-v{i},-v{j}', 1) for i, j in edges if road_status[(i, j)] == 'rainy']

        return {
            'clauses': clauses,
            'road_status': road_status,
            'future_weather': future_weather,
            'future_prob': future_prob,
            'capacities': capacities,
            'costs': costs
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        road_status = instance['road_status']
        future_weather = instance['future_weather']
        future_prob = instance['future_prob']
        capacities = instance['capacities']
        costs = instance['costs']

        model = Model("MaxSatisfiability")

        var_names = {}

        # Create variables for each literal and clause
        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var

        # Create integer variables for capacities and costs
        for (i, j), capacity in capacities.items():
            var_names[f"cap_{i}_{j}"] = model.addVar(vtype="I", lb=1, ub=capacity, name=f"cap_{i}_{j}")

        for (i, j), cost in costs.items():
            var_names[f"cost_{i}_{j}"] = model.addVar(vtype="I", lb=0, ub=cost, name=f"cost_{i}_{j}")

        # Objective function - maximize the number of satisfied clauses and minimize maintenance costs
        objective_expr = quicksum(
            var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        )

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

        # Modify the objective to include penalties for late deliveries
        for road, season in road_status.items():
            i, j = road
            penalty = model.addVar(vtype="B", name=f"penalty_{i}_{j}")
            if season == 'rainy':
                # Adding a penalty if the road is inaccessible due to season
                model.addCons(penalty >= 1, name=f"road_penalty_constraint_{i}_{j}")
                objective_expr += penalty * self.season_pen  # Penalty weight

        M = self.big_m  # Big M constant
        for (i, j), prob in zip(future_weather.keys(), future_prob):
            future_access = model.addVar(vtype="B", name=f"future_access_{i}_{j}")
            penalty_future = model.addVar(vtype="B", name=f"penalty_future_{i}_{j}")

            # Constraint for future accessibility based on weather predictions
            if future_weather[(i, j)] == 'rainy':
                model.addCons(future_access == 0, name=f"future_weather_{i}_{j}")
                model.addCons(penalty_future >= 1, name=f"future_penalty_{i}_{j}")
                objective_expr += penalty_future * prob * M  # Adjust penalty with future weather probability
            elif future_weather[(i, j)] == 'dry':
                model.addCons(future_access == 1, name=f"future_weather_{i}_{j}")

        # Include cost minimization in the objective function
        for (i, j), cost in costs.items():
            objective_expr -= var_names[f"cost_{i}_{j}"]

        ### Warehouse Layout Optimization Integration ###
        # Generate instance data for additional complexity
        n_facilities = 5  # Number of facilities
        n_units = 8  # Number of units
        min_transport_cost = 1
        max_transport_cost = 100
        min_facility_cost = 500
        max_facility_cost = 1000
        min_facility_space = 10
        max_facility_space = 20

        facility_costs = np.random.randint(min_facility_cost, max_facility_cost + 1, n_facilities)
        transport_costs = np.random.randint(min_transport_cost, max_transport_cost + 1, (n_facilities, n_units))
        spaces = np.random.randint(min_facility_space, max_facility_space + 1, n_facilities)
        demands = np.random.randint(1, 10, n_units)

        # Decision variables for facilities and transport
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}

        # Objective: Minimize facility and transport costs
        for f in range(n_facilities):
            objective_expr -= facility_costs[f] * facility_vars[f]
        for f in range(n_facilities):
            for u in range(n_units):
                objective_expr -= transport_costs[f, u] * transport_vars[f, u]
        
        # Constraints: Each unit demand is met by exactly one facility
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) == 1, f"Unit_{u}_Demand")

        # Constraints: Only open facilities can serve units
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f], f"Facility_{f}_Serve_{u}")

        # Constraints: Facilities cannot exceed their space
        for f in range(n_facilities):
            model.addCons(quicksum(demands[u] * transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 75,
        'max_n': 125,
        'er_prob': 0.5,
        'edge_addition_prob': 0.3,
        'season_pen': 10,
        'max_capacity': 100,
        'max_cost': 50,
        'big_m': 1000
    }

    maxsat = MaxSatisfiability(parameters, seed=seed)
    instance = maxsat.generate_instances()
    solve_status, solve_time = maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")