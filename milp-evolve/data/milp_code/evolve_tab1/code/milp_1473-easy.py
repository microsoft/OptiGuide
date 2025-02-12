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

    # Generate instance data considering seasonal weather and road status including future predictions
    def generate_instances(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_maxsat_graph(n)
        seasons = ['dry', 'rainy']
        road_status = {edge: random.choice(seasons) for edge in edges}
        future_weather = {edge: random.choice(seasons) for edge in edges}
        future_prob = np.random.rand(len(edges))
        clauses = [
            (f'v{i},v{j}', 1) for i, j in edges if road_status[(i, j)] == 'dry'
        ] + [(f'-v{i},-v{j}', 1) for i, j in edges if road_status[(i, j)] == 'rainy']
        
        road_weights = {edge: random.randint(1, 10) for edge in edges}
        capacities = random.randint(self.capacity_lb, self.capacity_ub)
        
        return {
            'clauses': clauses, 
            'road_status': road_status, 
            'future_weather': future_weather, 
            'future_prob': future_prob, 
            'road_weights': road_weights, 
            'capacities': capacities 
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        road_status = instance['road_status']
        future_weather = instance['future_weather']
        future_prob = instance['future_prob']
        road_weights = instance['road_weights']
        capacities = instance['capacities']

        model = Model("MaxSatisfiability")
        var_names = {}

        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var

        objective_expr = quicksum(
            var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        )

        for idx, (clause, weight) in enumerate(clauses):
            vars_in_clause = clause.split(',')
            clause_var = var_names[f"cl_{idx}"]
            positive_part = quicksum(var_names[var] for var in vars_in_clause if not var.startswith('-'))
            negative_part = quicksum(1 - var_names[var[1:]] for var in vars_in_clause if var.startswith('-'))
            total_satisfied = positive_part + negative_part
            if weight < np.inf:
                model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")
            else:
                model.addCons(total_satisfied >= 1, name=f"clause_{idx}")

        for road, season in road_status.items():
            i, j = road
            penalty = model.addVar(vtype="B", name=f"penalty_{i}_{j}")
            if season == 'rainy':
                model.addCons(penalty >= 1, name=f"road_penalty_constraint_{i}_{j}")
                objective_expr += penalty * self.season_pen

        ### New Set Packing and Set Covering Constraints start here
        M = int(self.big_m)  # Big M constant
        set_packing_expr = []
        road_vars = {}

        for (i, j), prob in zip(future_weather.keys(), future_prob):
            future_access = model.addVar(vtype="B", name=f"future_access_{i}_{j}")
            penalty_future = model.addVar(vtype="B", name=f"penalty_future_{i}_{j}")
            road_var = model.addVar(vtype="B", name=f"road_{i}_{j}")
            road_vars[(i, j)] = road_var

            if future_weather[(i, j)] == 'rainy':
                model.addCons(future_access == 0, name=f"future_weather_{i}_{j}")
                model.addCons(penalty_future >= 1, name=f"future_penalty_{i}_{j}")
                model.addCons(future_access <= penalty_future, name=f"ch_access_penalty_{i}_{j}")
                model.addCons(penalty_future <= M * (1 - future_access), name=f"ch_penalty_future_{i}_{j}")
                objective_expr += penalty_future * prob * M
            
            elif future_weather[(i, j)] == 'dry':
                model.addCons(future_access == 1, name=f"future_weather_{i}_{j}")

            set_packing_expr.append(road_var)

        # Set Packing: Ensure certain roads are included in the solution only once
        model.addCons(quicksum(road_vars[road] for road in road_vars) <= capacities, name="set_packing")

        ### New Set Covering constraints
        for road in road_vars:
            if road_status[road] == 'dry':
                model.addCons(quicksum(road_vars[road] for road in road_vars if road_status[road] == 'dry') >= 1, name=f"set_covering_{road}")

        ### New constraint code ends here
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 112,
        'max_n': 125,
        'er_prob': 0.17,
        'edge_addition_prob': 0.66,
        'season_pen': 5,
        'big_m': 562,
        'capacity_lb': 3,
        'capacity_ub': 135,
    }

    maxsat = MaxSatisfiability(parameters, seed=seed)
    instance = maxsat.generate_instances()
    solve_status, solve_time = maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")