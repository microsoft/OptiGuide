import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class PharmaSupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_supplies > 0 and self.n_centers > 0 and self.n_regions > 0
        assert self.min_prod_cost >= 0 and self.max_prod_cost >= self.min_prod_cost
        assert self.min_trans_cost >= 0 and self.max_trans_cost >= self.min_trans_cost
        assert self.min_demand > 0 and self.max_demand >= self.min_demand
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        production_costs = np.random.randint(self.min_prod_cost, self.max_prod_cost + 1, self.n_supplies)
        transportation_costs = np.random.randint(self.min_trans_cost, self.max_trans_cost + 1, (self.n_centers, self.n_regions, self.n_supplies))
        region_demands = np.random.randint(self.min_demand, self.max_demand + 1, (self.n_regions, self.n_supplies))
        center_capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_centers)
        supply_shelf_life = np.random.randint(1, 10, self.n_supplies)  # In days
        penalty_costs = np.random.uniform(50, 150, self.n_regions * self.n_supplies).reshape(self.n_regions, self.n_supplies)
        
        return {
            "production_costs": production_costs,
            "transportation_costs": transportation_costs,
            "region_demands": region_demands,
            "center_capacities": center_capacities,
            "supply_shelf_life": supply_shelf_life,
            "penalty_costs": penalty_costs,
        }

    def solve(self, instance):
        production_costs = instance['production_costs']
        transportation_costs = instance['transportation_costs']
        region_demands = instance['region_demands']
        center_capacities = instance['center_capacities']
        supply_shelf_life = instance['supply_shelf_life']
        penalty_costs = instance['penalty_costs']
        
        model = Model("PharmaSupplyChainOptimization")
        n_supplies = len(production_costs)
        n_centers = len(center_capacities)
        n_regions = len(region_demands)
        
        production_vars = {(s): model.addVar(vtype="I", name=f"Prod_Supply_{s}") for s in range(n_supplies)}
        distribution_vars = {(c, r, s): model.addVar(vtype="I", name=f"Dist_Center_{c}_Region_{r}_Supply_{s}") for c in range(n_centers) for r in range(n_regions) for s in range(n_supplies)}
        unmet_demand_vars = {(r, s): model.addVar(vtype="I", name=f"Unmet_Region_{r}_Supply_{s}") for r in range(n_regions) for s in range(n_supplies)}

        # Objective Function
        model.setObjective(
            quicksum(production_costs[s] * production_vars[s] for s in range(n_supplies)) +
            quicksum(transportation_costs[c][r][s] * distribution_vars[c, r, s] for c in range(n_centers) for r in range(n_regions) for s in range(n_supplies)) +
            quicksum(penalty_costs[r][s] * unmet_demand_vars[r, s] for r in range(n_regions) for s in range(n_supplies)),
            "minimize"
        )

        # Constraints
        for r in range(n_regions):
            for s in range(n_supplies):
                model.addCons(
                    quicksum(distribution_vars[c, r, s] for c in range(n_centers)) + unmet_demand_vars[r, s] == region_demands[r][s],
                    f"Demand_Fulfillment_Region_{r}_Supply_{s}"
                )

        for c in range(n_centers):
            for s in range(n_supplies):
                model.addCons(
                    quicksum(distribution_vars[c, r, s] for r in range(n_regions)) <= production_vars[s],
                    f"Distribution_Limit_Center_{c}_Supply_{s}"
                )
        
        for c in range(n_centers):
            model.addCons(
                quicksum(distribution_vars[c, r, s] for r in range(n_regions) for s in range(n_supplies)) <= center_capacities[c],
                f"Center_Capacity_{c}"
            )
        
        for s in range(n_supplies):
            model.addCons(
                production_vars[s] <= supply_shelf_life[s] * 100,
                f"Shelf_Life_Constraint_Supply_{s}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_supplies': 40,
        'n_centers': 80,
        'n_regions': 10,
        'min_prod_cost': 200,
        'max_prod_cost': 700,
        'min_trans_cost': 40,
        'max_trans_cost': 350,
        'min_demand': 100,
        'max_demand': 3000,
        'min_capacity': 2000,
        'max_capacity': 5000,
    }

    pharma_optimizer = PharmaSupplyChainOptimization(parameters, seed=42)
    instance = pharma_optimizer.generate_instance()
    solve_status, solve_time, objective_value = pharma_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")