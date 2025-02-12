import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ProjectPortfolioOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0
        assert self.n_projects > 0
        assert self.n_periods > 0
        assert self.min_opening_cost >= 0 and self.max_opening_cost >= self.min_opening_cost
        assert self.min_project_utility >= 0 and self.max_project_utility >= self.min_project_utility

        opening_costs = np.random.randint(self.min_opening_cost, self.max_opening_cost + 1, self.n_facilities)
        project_utilities = np.random.randint(self.min_project_utility, self.max_project_utility + 1, self.n_projects)
        period_durations = np.random.randint(self.min_period_duration, self.max_period_duration + 1, (self.n_projects, self.n_periods))
        transport_costs = np.random.uniform(self.min_transport_cost, self.max_transport_cost, (self.n_facilities, self.n_projects))

        return {
            "opening_costs": opening_costs,
            "project_utilities": project_utilities,
            "period_durations": period_durations,
            "transport_costs": transport_costs,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        opening_costs = instance['opening_costs']
        project_utilities = instance['project_utilities']
        period_durations = instance['period_durations']
        transport_costs = instance['transport_costs']
        
        model = Model("ProjectPortfolioOptimization")
        n_facilities = len(opening_costs)
        n_projects = len(project_utilities)
        n_periods = len(period_durations[0])
        
        # Decision variables
        FacilityVars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        AllocationVars = {(f, p): model.addVar(vtype="B", name=f"Facility_{f}_Project_{p}") for f in range(n_facilities) for p in range(n_projects)}
        TimeAllocationVars = {(p, t): model.addVar(vtype="B", name=f"Project_{p}_Period_{t}") for p in range(n_projects) for t in range(n_periods)}
        
        # Objective: maximize utility minus costs, considering transport and facility opening costs
        model.setObjective(
            quicksum(project_utilities[p] * TimeAllocationVars[p, t] for p in range(n_projects) for t in range(n_periods)) -
            quicksum(opening_costs[f] * FacilityVars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][p] * AllocationVars[f, p] for f in range(n_facilities) for p in range(n_projects)),
            "maximize"
        )
        
        # Constraints: Each project must be allocated to at least one facility and one period
        for p in range(n_projects):
            model.addCons(quicksum(AllocationVars[f, p] for f in range(n_facilities)) == 1, f"Project_{p}_Allocation")
            model.addCons(quicksum(TimeAllocationVars[p, t] for t in range(n_periods)) == 1, f"Project_{p}_Time_Allocation")
        
        # Constraints: Only open facilities can allocate projects
        for f in range(n_facilities):
            for p in range(n_projects):
                model.addCons(AllocationVars[f, p] <= FacilityVars[f], f"Facility_{f}_Project_{p}_Service")
        
        # Constraints: Total time allocation for each period should not exceed limits
        for t in range(n_periods):
            model.addCons(quicksum(TimeAllocationVars[p, t] * period_durations[p][t] for p in range(n_projects)) <= self.max_period_duration, f"Period_{t}_Limit")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 40,
        'n_projects': 200,
        'n_periods': 32,
        'min_opening_cost': 1500,
        'max_opening_cost': 2000,
        'min_project_utility': 2000,
        'max_project_utility': 2500,
        'min_period_duration': 5,
        'max_period_duration': 135,
        'min_transport_cost': 10,
        'max_transport_cost': 50,
    }
    
    portfolio_optimizer = ProjectPortfolioOptimization(parameters, seed=seed)
    instance = portfolio_optimizer.generate_instance()
    solve_status, solve_time, objective_value = portfolio_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")