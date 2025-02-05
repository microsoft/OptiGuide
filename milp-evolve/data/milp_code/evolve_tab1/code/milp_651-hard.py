import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class CapacitatedFacilityLocationWithMultipliers:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2 +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2
        )
        return costs

    def generate_instance(self):
        demands_per_period = {t: self.randint(self.n_customers, self.demand_interval) for t in range(self.n_periods)}
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_facilities, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_facilities, self.fixed_cost_cste_interval)
        )
        transportation_costs = self.unit_transportation_costs()
        
        capacities = capacities * self.ratio * np.sum(list(demands_per_period.values())) / self.n_periods / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands_per_period': demands_per_period,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        # Environmental impact costs for each facility
        env_impact_costs = self.randint((self.n_facilities, self.n_periods), self.env_impact_cost_interval)

        # Random capacity disruptions, simulating stochastic disruptions
        disruption_impact_factor = self.randint(self.n_facilities, self.disruption_impact_interval)
        disruption_periods = self.randint(self.n_periods, self.disruption_periods_interval)

        res.update({
            'env_impact_costs': env_impact_costs,
            'disruption_impact_factor': disruption_impact_factor,
            'disruption_periods': disruption_periods
        })

        # Multi-period investment return factors
        investment_returns = self.randint((self.n_facilities, self.n_periods), self.investment_return_interval)
        res['investment_returns'] = investment_returns

        ### New instance data code ends here
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands_per_period = instance['demands_per_period']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        env_impact_costs = instance['env_impact_costs']
        disruption_impact_factor = instance['disruption_impact_factor']
        disruption_periods = instance['disruption_periods']
        investment_returns = instance['investment_returns']
        
        n_customers = len(demands_per_period[0])
        n_facilities = len(capacities)
        n_periods = len(demands_per_period)
        
        model = Model("MultiPeriodFacilityLocation")
        
        # Decision variables
        open_facilities = {
            (j, t): model.addVar(vtype="B", name=f"Open_{j}_{t}")
            for j in range(n_facilities)
            for t in range(n_periods)
        }
        serve = {
            (i, j, t): model.addVar(vtype="C", name=f"Serve_{i}_{j}_{t}")
            for i in range(n_customers)
            for j in range(n_facilities)
            for t in range(n_periods)
        }
        
        # New variable: investment multiplier for scaling effect
        investment_multiplier = {
            (j, t): model.addVar(vtype="C", name=f"InvestmentMultiplier_{j}_{t}")
            for j in range(n_facilities)
            for t in range(n_periods)
        }
        
        # Objective: minimize the total operating and environmental cost minus investment returns
        primary_objective_expr = quicksum(
            fixed_costs[j] * open_facilities[j, t] 
            + transportation_costs[i, j] * serve[i, j, t] 
            + env_impact_costs[j, t] * open_facilities[j, t] 
            for i in range(n_customers)
            for j in range(n_facilities)
            for t in range(n_periods)
        )
        
        investment_return_expr = quicksum(
            investment_returns[j, t] * investment_multiplier[j, t]
            for j in range(n_facilities)
            for t in range(n_periods)
        )
        
        model.setObjective(primary_objective_expr - investment_return_expr, "minimize")

        # Constraints: demand must be met in each period
        for i in range(n_customers):
            for t in range(n_periods):
                model.addCons(
                    quicksum(serve[i, j, t] for j in range(n_facilities)) >= demands_per_period[t][i],
                    f"Demand_{i}_{t}"
                )
        
        # Constraints: capacity limits in each period with potential disruption impact
        for j in range(n_facilities):
            for t in range(n_periods):
                capacity_factor = capacities[j] * (1 - disruption_impact_factor[j]/100 if t in disruption_periods else 1)
                model.addCons(
                    quicksum(serve[i, j, t] for i in range(n_customers)) <= capacity_factor * open_facilities[j, t],
                    f"Capacity_{j}_{t}"
                )

                # Investment scaling effect
                model.addCons(
                    investment_multiplier[j, t] <= open_facilities[j, t],
                    f"InvestmentMultiplier_{j}_{t}"
                )

        # Constraints: phased investment or hierarchical opening constraints
        for j in range(n_facilities):
            for t in range(1, n_periods):
                model.addCons(
                    open_facilities[j, t] >= open_facilities[j, t-1],
                    f"PhasedInvestment_{j}_{t}"
                )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 100,
        'n_facilities': 50,
        'n_periods': 5,
        'demand_interval': (5, 36),
        'capacity_interval': (10, 161),
        'fixed_cost_scale_interval': (100, 111),
        'fixed_cost_cste_interval': (0, 91),
        'ratio': 5.0,
        'env_impact_cost_interval': (50, 200),
        'disruption_impact_interval': (5, 20),
        'disruption_periods_interval': (1, 3),
        'investment_return_interval': (10, 50)
    }
    
    facility_location = CapacitatedFacilityLocationWithMultipliers(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")