import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
class DemandCoverageGraph:
    def __init__(self, facilities, demand_points, coverage, cost):
        self.facilities = facilities
        self.demand_points = demand_points
        self.coverage = coverage  # dict of demand point to sets of covering facilities
        self.cost = cost  # dict of facilities to their opening costs

    @staticmethod
    def generate(number_of_facilities, number_of_demand_points, coverage_probability, cost_range):
        facilities = np.arange(number_of_facilities)
        demand_points = np.arange(number_of_demand_points)
        coverage = {dp: set() for dp in demand_points}
        cost = {fac: random.randint(*cost_range) for fac in facilities}

        for dp in demand_points:
            for fac in facilities:
                if np.random.uniform() < coverage_probability:
                    coverage[dp].add(fac)
                    
        return DemandCoverageGraph(facilities, demand_points, coverage, cost)

############# Helper function #############

class MaximumCoverageProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        graph = DemandCoverageGraph.generate(self.n_facilities, self.n_demand_points, self.coverage_probability, self.cost_range)
        res = {'graph': graph}
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        model = Model("MaximumCoverageProblem")
        facility_open = {}
        demand_covered = {}

        for fac in graph.facilities:
            facility_open[fac] = model.addVar(vtype="B", name=f"facility_open_{fac}")

        for dp in graph.demand_points:
            demand_covered[dp] = model.addVar(vtype="B", name=f"demand_covered_{dp}")

        for dp in graph.demand_points:
            model.addCons(quicksum(facility_open[fac] for fac in graph.coverage[dp]) >= demand_covered[dp], name=f"coverage_constraint_{dp}")

        total_cost = quicksum(graph.cost[fac] * facility_open[fac] for fac in graph.facilities)
        model.addCons(total_cost <= self.budget_limit, name="budget_constraint")

        objective_expr = quicksum(demand_covered[dp] for dp in graph.demand_points)
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 600,
        'n_demand_points': 500,
        'coverage_probability': 0.1,
        'cost_range': (9, 900),
        'budget_limit': 5000,
    }

    max_coverage_problem = MaximumCoverageProblem(parameters, seed=seed)
    instance = max_coverage_problem.generate_instance()
    solve_status, solve_time = max_coverage_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")