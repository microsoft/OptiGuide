import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NeighborhoodGuardPatrolProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_graph(self, n):
        return nx.erdos_renyi_graph(n, self.er_prob, seed=self.seed)

    def get_instance(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        G = self.generate_graph(n)
        num_high_priority_buildings = n // 3
        num_neighborhoods = n - num_high_priority_buildings
        high_priority_buildings = random.sample(range(n), num_high_priority_buildings)
        neighborhoods = list(set(range(n)) - set(high_priority_buildings))

        patrol_fees = {i: np.random.randint(500, 5000) for i in neighborhoods}
        patrol_times = {f"{i}_{j}": np.random.randint(1, 10) for i in high_priority_buildings for j in neighborhoods}

        res = {
            'high_priority_buildings': high_priority_buildings,
            'neighborhoods': neighborhoods,
            'patrol_fees': patrol_fees,
            'patrol_times': patrol_times
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        high_priority_buildings = instance['high_priority_buildings']
        neighborhoods = instance['neighborhoods']
        patrol_fees = instance['patrol_fees']
        patrol_times = instance['patrol_times']

        model = Model("NeighborhoodGuardPatrol")
        patrol_vars = {i: model.addVar(vtype="B", name=f"patrol_{i}") for i in neighborhoods}
        coverage_vars = {f"{i}_{j}": model.addVar(vtype="B", name=f"cover_{i}_{j}")
                         for i in high_priority_buildings for j in neighborhoods}

        # Objective function: Minimize total patrol fees while maximizing coverage
        patrol_fee_expr = quicksum(patrol_fees[j] * patrol_vars[j] for j in neighborhoods)
        max_coverage_expr = quicksum(coverage_vars[f"{i}_{j}"]
                                     for i in high_priority_buildings for j in neighborhoods)
        model.setObjective(patrol_fee_expr - max_coverage_expr, "maximize")

        # Constraint 1: Each high-priority building must be covered by at least one patrol.
        for i in high_priority_buildings:
            model.addCons(quicksum(coverage_vars[f"{i}_{j}"] for j in neighborhoods) >= 1, 
                          name=f"coverage_{i}")

        # Constraint 2: High-priority building i can only be covered by a patrol assigned to neighborhood j.
        for i in high_priority_buildings:
            for j in neighborhoods:
                model.addCons(coverage_vars[f"{i}_{j}"] <= patrol_vars[j], name=f"cover_cond_{i}_{j}")

        # Constraint 3: Total patrol fees should not exceed MaxPatrolBudget.
        model.addCons(quicksum(patrol_fees[j] * patrol_vars[j] for j in neighborhoods) <= self.MaxPatrolBudget, 
                      name="patrol_budget")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 150,
        'max_n': 500,
        'er_prob': 0.74,
        'MaxPatrolBudget': 20000,
    }

    patrol_problem = NeighborhoodGuardPatrolProblem(parameters, seed=seed)
    instance = patrol_problem.get_instance()
    solve_status, solve_time = patrol_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")