import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum, multidict

class FarmersMarketPlacement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.num_markets > 0 and self.num_customers > 0
        assert self.min_management_cost >= 0 and self.max_management_cost >= self.min_management_cost
        assert self.min_customer_cost >= 0 and self.max_customer_cost >= self.min_customer_cost
        assert self.min_market_coverage > 0 and self.max_market_coverage >= self.min_market_coverage

        # Simplified data generation for management and customer costs
        management_costs = np.random.randint(self.min_management_cost, self.max_management_cost + 1, self.num_markets)
        customer_costs = np.random.randint(self.min_customer_cost, self.max_customer_cost + 1, (self.num_markets, self.num_customers))
        coverages = np.random.randint(self.min_market_coverage, self.max_market_coverage + 1, self.num_markets)
        demands = np.random.randint(1, 10, self.num_customers)

        return {
            "management_costs": management_costs,
            "customer_costs": customer_costs,
            "coverages": coverages,
            "demands": demands,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        management_costs = instance['management_costs']
        customer_costs = instance['customer_costs']
        coverages = instance['coverages']
        demands = instance['demands']
        
        model = Model("FarmersMarketPlacement")
        num_markets = len(management_costs)
        num_customers = len(customer_costs[0])
        
        # Decision variables
        market_open = {m: model.addVar(vtype="B", name=f"MarketOpen_{m}") for m in range(num_markets)}
        customer_covered = {(m, r): model.addVar(vtype="B", name=f"Market_{m}_Customer_{r}") for m in range(num_markets) for r in range(num_customers)}

        # Objective: minimize the total management cost including market rents and transportation costs
        model.setObjective(
            quicksum(management_costs[m] * market_open[m] for m in range(num_markets)) +
            quicksum(customer_costs[m, r] * customer_covered[m, r] for m in range(num_markets) for r in range(num_customers)), 
            "minimize"
        )

        # Constraints: Total management cost can't exceed the given budget
        model.addCons(quicksum(management_costs[m] * market_open[m] for m in range(num_markets)) <= self.budget_constraint, "TotalCostBudget")
        
        # Constraints: Only open market stands can cover customers
        for m in range(num_markets):
            for r in range(num_customers):
                model.addCons(customer_covered[m, r] <= market_open[m], f"Market_{m}_Cover_{r}")

        # Constraints: Capacity Limits of Markets
        for m in range(num_markets):
            model.addCons(quicksum(customer_covered[m, r] * demands[r] for r in range(num_customers)) <= coverages[m], f"Market_{m}_Capacity")
        
        # Constraints: Total Market Coverage meets Total Demand
        total_demand = np.sum(demands)
        model.addCons(quicksum(coverages[m] * market_open[m] for m in range(num_markets)) >= total_demand, "TotalDemandCover")

        # New Constraints: Set Covering constraints to ensure each customer is covered by at least one market
        for r in range(num_customers):
            model.addCons(quicksum(customer_covered[m, r] for m in range(num_markets)) >= 1, f"Customer_{r}_Coverage")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_markets': 150,
        'num_customers': 56,
        'min_customer_cost': 3000,
        'max_customer_cost': 5000,
        'min_management_cost': 1264,
        'max_management_cost': 3000,
        'min_market_coverage': 9,
        'max_market_coverage': 180,
        'budget_constraint': 25000,
    }

    market_optimizer = FarmersMarketPlacement(parameters, seed)
    instance = market_optimizer.generate_instance()
    solve_status, solve_time, objective_value = market_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")