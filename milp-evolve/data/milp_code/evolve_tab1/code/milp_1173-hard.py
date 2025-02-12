import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

class UrbanParkPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_parks > 0 and self.n_plots > 0
        assert self.min_park_cost >= 0 and self.max_park_cost >= self.min_park_cost
        assert self.min_plot_cost >= 0 and self.max_plot_cost >= self.min_plot_cost
        assert self.min_park_area > 0 and self.max_park_area >= self.min_park_area

        park_costs = np.random.randint(self.min_park_cost, self.max_park_cost + 1, self.n_parks)
        plot_costs = np.random.randint(self.min_plot_cost, self.max_plot_cost + 1, (self.n_parks, self.n_plots))
        areas = np.random.randint(self.min_park_area, self.max_park_area + 1, self.n_parks)
        demands = np.random.randint(1, 10, self.n_plots)
        
        return {
            "park_costs": park_costs,
            "plot_costs": plot_costs,
            "areas": areas,
            "demands": demands,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        park_costs = instance['park_costs']
        plot_costs = instance['plot_costs']
        areas = instance['areas']
        demands = instance['demands']
        
        model = Model("UrbanParkPlanning")
        n_parks = len(park_costs)
        n_plots = len(plot_costs[0])
        
        # Decision variables
        park_vars = {p: model.addVar(vtype="B", name=f"Park_{p}") for p in range(n_parks)}
        plot_vars = {(p, t): model.addVar(vtype="B", name=f"Park_{p}_Plot_{t}") for p in range(n_parks) for t in range(n_plots)}

        # Objective: minimize the total cost including park costs and plot costs
        model.setObjective(
            quicksum(park_costs[p] * park_vars[p] for p in range(n_parks)) +
            quicksum(plot_costs[p, t] * plot_vars[p, t] for p in range(n_parks) for t in range(n_plots)), "minimize"
        )
        
        # Constraints: Each plot demand is met by exactly one park
        for t in range(n_plots):
            model.addCons(quicksum(plot_vars[p, t] for p in range(n_parks)) == 1, f"Plot_{t}_Demand")
        
        # Constraints: Only open parks can serve plots
        for p in range(n_parks):
            for t in range(n_plots):
                model.addCons(plot_vars[p, t] <= park_vars[p], f"Park_{p}_Serve_{t}")
        
        # Constraints: Parks cannot exceed their area
        for p in range(n_parks):
            model.addCons(quicksum(demands[t] * plot_vars[p, t] for t in range(n_plots)) <= areas[p], f"Park_{p}_Area")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_parks': 40,
        'n_plots': 150,
        'min_plot_cost': 875,
        'max_plot_cost': 1000,
        'min_park_cost': 2000,
        'max_park_cost': 2000,
        'min_park_area': 100,
        'max_park_area': 2000,
    }

    park_planning_optimizer = UrbanParkPlanning(parameters, seed=42)
    instance = park_planning_optimizer.generate_instance()
    solve_status, solve_time, objective_value = park_planning_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")