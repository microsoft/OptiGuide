import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CapacitatedFacilityLocationSimplified:
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

        # Normalize the capacities
        capacities = capacities * self.ratio * np.sum(list(demands_per_period.values())) / self.n_periods / np.sum(capacities)
        capacities = np.round(capacities)

        res = {
            'demands_per_period': demands_per_period,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        # New data for set covering
        res['customer_subset_coverage'] = {
            t: {c: np.random.choice(self.n_facility_subsets, size=self.facility_subsets_to_covers, replace=False)
                for c in range(self.n_customer_subsets)} for t in range(self.n_periods)
        }

        # New data for distances and speed limits
        res['distances'] = np.random.rand(self.n_customers, self.n_facilities) * 50  # Distances between 0 and 50 units
        res['speed_limits'] = np.random.randint(self.speed_limit_interval[0], self.speed_limit_interval[1], (self.n_customers, self.n_facilities))  # Speed limits
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands_per_period = instance['demands_per_period']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        customer_subset_coverage = instance['customer_subset_coverage']
        distances = instance['distances']
        speed_limits = instance['speed_limits']

        n_customers = len(demands_per_period[0])
        n_facilities = len(capacities)
        n_periods = len(demands_per_period)
        n_customer_subsets = self.n_customer_subsets
        n_facility_subsets = self.n_facility_subsets

        model = Model("SimplifiedFacilityLocation")

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
        travel_time = {
            (i, j, t): model.addVar(vtype="C", name=f"TravelTime_{i}_{j}_{t}")
            for i in range(n_customers)
            for j in range(n_facilities)
            for t in range(n_periods)
        }

        # Objective: minimize the total fixed and transportation costs
        model.setObjective(
            quicksum(
                fixed_costs[j] * open_facilities[j, t] +
                transportation_costs[i, j] * serve[i, j, t]
                for i in range(n_customers)
                for j in range(n_facilities)
                for t in range(n_periods)
            ), "minimize"
        )

        # Constraints: demand must be met in each period
        for i in range(n_customers):
            for t in range(n_periods):
                model.addCons(
                    quicksum(serve[i, j, t] for j in range(n_facilities)) >= demands_per_period[t][i],
                    f"Demand_{i}_{t}"
                )

        # Constraints: capacity limits in each period
        for j in range(n_facilities):
            for t in range(n_periods):
                model.addCons(
                    quicksum(serve[i, j, t] for i in range(n_customers)) <= capacities[j] * open_facilities[j, t],
                    f"Capacity_{j}_{t}"
                )

        # Constraints: phased investments (facilities can only open once and stay open)
        for j in range(n_facilities):
            for t in range(1, n_periods):
                model.addCons(
                    open_facilities[j, t] >= open_facilities[j, t-1],
                    f"PhasedInvestment_{j}_{t}"
                )

        # New constraints for set covering
        # A customer subset must be served by at least one facility subset
        for t in range(n_periods):
            for c in range(n_customer_subsets):
                model.addCons(
                    quicksum(open_facilities[j, t] for j in customer_subset_coverage[t][c]) >= 1,
                    f"CustomerSubsetCover_{c}_{t}"
                )

        # New constraints for travel times based on speed limits
        for i in range(n_customers):
            for j in range(n_facilities):
                for t in range(n_periods):
                    model.addCons(
                        travel_time[i, j, t] == distances[i, j] / speed_limits[i, j],
                        f"TravelTimeConstraint_{i}_{j}_{t}"
                    )
                    model.addCons(
                        travel_time[i, j, t] <= self.max_delivery_time,
                        f"DeliveryTimeWindow_{i}_{j}_{t}"
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
        'n_customer_subsets': 10,
        'n_facility_subsets': 5,
        'facility_subsets_to_covers': 2,
        'speed_limit_interval': (30, 70),
        'max_delivery_time': 2.0
    }

    facility_location = CapacitatedFacilityLocationSimplified(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")