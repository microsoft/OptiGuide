import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

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

    # Generate instance data considering seasonal weather and road status
    def generate_instances(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_maxsat_graph(n)

        # Randomly generate seasons affecting road conditions
        seasons = ['dry', 'rainy']
        road_status = {edge: random.choice(seasons) for edge in edges}

        # Generate clauses considering road conditions
        clauses = [
            (f'v{i},v{j}', 1) for i, j in edges if road_status[(i, j)] == 'dry'
        ] + [(f'-v{i},-v{j}', 1) for i, j in edges if road_status[(i, j)] == 'rainy']

        # Generate warehouse capacities randomly
        warehouse_capacities = {f'city_{i}': np.random.randint(5, 20) for i in range(n)}

        # Generate transportation modes with costs and capacities
        transport_modes = ['truck', 'ship', 'airplane']
        transport_costs = {mode: np.random.randint(1, 10) for mode in transport_modes}
        transport_caps = {mode: np.random.randint(10, 100) for mode in transport_modes}

        # Generate priority lists based on proximity, randomly
        city_priority = {f'city_{i}': np.random.choice(range(n), size=5, replace=False).tolist() for i in range(n)}

        # Generate time windows for delivery schedules
        delivery_schedules = {f'city_{i}': (np.random.randint(0, 10), np.random.randint(10, 20)) for i in range(n)}

        return {
            'clauses': clauses,
            'road_status': road_status,
            'warehouse_capacities': warehouse_capacities,
            'transport_costs': transport_costs,
            'transport_caps': transport_caps,
            'city_priority': city_priority,
            'delivery_schedules': delivery_schedules
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        road_status = instance['road_status']
        warehouse_capacities = instance['warehouse_capacities']
        transport_costs = instance['transport_costs']
        transport_caps = instance['transport_caps']
        city_priority = instance['city_priority']
        delivery_schedules = instance['delivery_schedules']

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

        # Objective function - maximize the number of satisfied clauses
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

        # Modify the objective to include penalties for rainy seasons
        for road, season in road_status.items():
            i, j = road
            penalty = model.addVar(vtype="B", name=f"penalty_{i}_{j}")
            if season == 'rainy':
                # Adding a penalty if the road is inaccessible due to season
                model.addCons(penalty >= 1, name=f"road_penalty_constraint_{i}_{j}")
                objective_expr += penalty * self.season_pen  # Penalty weight

        # Additional constraints

        # Warehouse capacity constraints
        for city, capacity in warehouse_capacities.items():
            warehouse_var = model.addVar(vtype="B", name=f"warehouse_{city}")
            model.addCons(warehouse_var <= capacity, name=f"warehouse_capacity_constraint_{city}")

        # Transportation cost and capacity constraints
        for mode, cost in transport_costs.items():
            cap = transport_caps[mode]
            transport_var = model.addVar(vtype="I", name=f"transport_{mode}")
            model.addCons(transport_var <= cap, name=f"transport_capacity_constraint_{mode}")
            objective_expr -= cost * transport_var  # Minimize cost by subtracting

        # Inter-city aid priority constraints
        for city, priority_list in city_priority.items():
            for priority_city in priority_list:
                priority_var = model.addVar(vtype="B", name=f"priority_{city}_to_{priority_city}")
                model.addCons(priority_var <= 1, name=f"priority_constraint_{city}_to_{priority_city}")

        # Time-window coordination constraints
        for city, (start, end) in delivery_schedules.items():
            time_var = model.addVar(vtype="I", name=f"time_{city}")
            model.addCons(time_var >= start, name=f"time_window_start_constraint_{city}")
            model.addCons(time_var <= end, name=f"time_window_end_constraint_{city}")

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
        'season_pen': 10  # Seasonal penalty weight
    }

    maxsat = MaxSatisfiability(parameters, seed=seed)
    instance = maxsat.generate_instances()
    solve_status, solve_time = maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")