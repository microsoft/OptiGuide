import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ComplexFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
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
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_facilities, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_facilities, self.fixed_cost_cste_interval)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        # Additional data from the second MILP
        n_routes = min(self.max_routes, self.n_customers * self.n_facilities)
        route_costs = []
        package_weights = []

        while len(route_costs) < n_routes:
            path_length = np.random.randint(1, self.max_route_length + 1)
            path = np.random.choice(self.n_customers, size=path_length, replace=False)
            cost = max(np.sum(demands[path]) + np.random.normal(0, 10), 0)
            complexity = np.random.poisson(lam=5)

            route_costs.append((path.tolist(), cost, complexity))

        overload_penalty_coeff = 100 * np.random.rand()
        high_demand_bonus_threshold = 90  # percentile value for demand

        res.update({
            "route_costs": route_costs,
            "overload_penalty_coeff": overload_penalty_coeff,
            "high_demand_bonus_threshold": high_demand_bonus_threshold,
        })

        return res

    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        route_costs = instance['route_costs']
        overload_penalty_coeff = instance['overload_penalty_coeff']
        high_demand_bonus_threshold = instance['high_demand_bonus_threshold']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("ComplexFacilityLocation")
        
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        penalty_vars = {j: model.addVar(vtype="I", name=f"Penalty_{j}") for j in range(n_facilities)}
        bonus_vars = {(i, j): model.addVar(vtype="B", name=f"HighDemandBonus_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # Primary objective: minimize total cost
        primary_objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) \
                         + quicksum(overload_penalty_coeff * penalty_vars[j] for j in range(n_facilities)) \
                         - quicksum(high_demand_bonus_threshold * bonus_vars[i, j] for i in range(n_customers) for j in range(n_facilities))

        model.setObjective(primary_objective_expr, "minimize")

        # Constraint: Each customer must be served by at least one facility
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}")
        
        # Constraint: Capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        # Constraint: Total demand must be less than or equal to total capacity
        total_demand = np.sum(demands)
        model.addCons(quicksum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_demand, "TotalDemand")

        # Constraint: Serve only from open facilities
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(serve[i, j] <= open_facilities[j], f"Tightening_{i}_{j}")

        # Constraint: Penalty for exceeding capacity
        for j in range(n_facilities):
            model.addCons(penalty_vars[j] >= 0.1 * (quicksum(serve[i, j] * demands[i] for i in range(n_customers)) - capacities[j]), f"PenaltyCap_{j}")
        
        # Constraint: Bonus for serving high demand customers
        high_demand_threshold = np.percentile(demands, 90)
        for i in range(n_customers):
            for j in range(n_facilities):
                if demands[i] > high_demand_threshold:
                    model.addCons(bonus_vars[i, j] <= serve[i, j], f"HighDemandBonus_{i}_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 25,
        'n_facilities': 600,
        'demand_interval': (10, 50),
        'capacity_interval': (120, 600),
        'fixed_cost_scale_interval': (150, 750),
        'fixed_cost_cste_interval': (0, 5),
        'ratio': 30.0,
        'max_routes': 1200,
        'max_route_length': 7,
    }
    
    facility_location = ComplexFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")