import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CallCenterOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_call_centers > 0 and self.n_regions > 0
        assert self.min_center_cost >= 0 and self.max_center_cost >= self.min_center_cost
        assert self.min_contact_cost >= 0 and self.max_contact_cost >= self.min_contact_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        center_costs = np.random.randint(self.min_center_cost, self.max_center_cost + 1, self.n_call_centers)
        contact_costs = np.random.randint(self.min_contact_cost, self.max_contact_cost + 1, (self.n_call_centers, self.n_regions))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_call_centers)
        region_demand = np.random.randint(1, 20, self.n_regions)
        operational_limits = np.random.uniform(self.min_operational_limit, self.max_operational_limit, self.n_call_centers)
        time_windows = np.random.uniform(self.min_time_window, self.max_time_window, self.n_call_centers)
        
        return {
            "center_costs": center_costs,
            "contact_costs": contact_costs,
            "capacities": capacities,
            "region_demand": region_demand,
            "operational_limits": operational_limits,
            "time_windows": time_windows,
        }

    def solve(self, instance):
        center_costs = instance['center_costs']
        contact_costs = instance['contact_costs']
        capacities = instance['capacities']
        region_demand = instance['region_demand']
        operational_limits = instance['operational_limits']
        time_windows = instance['time_windows']

        model = Model("CallCenterOptimization")
        n_call_centers = len(center_costs)
        n_regions = len(contact_costs[0])

        # Decision variables
        open_vars = {c: model.addVar(vtype="B", name=f"Center_{c}") for c in range(n_call_centers)}
        serve_vars = {(c, r): model.addVar(vtype="B", name=f"Serve_{c}_{r}") for c in range(n_call_centers) for r in range(n_regions)}
        handle_vars = {(c, r): model.addVar(vtype="C", name=f"Handle_{c}_{r}") for c in range(n_call_centers) for r in range(n_regions)}

        # Objective: minimize the total cost including call center operating costs and contact costs.
        model.setObjective(
            quicksum(center_costs[c] * open_vars[c] for c in range(n_call_centers)) +
            quicksum(contact_costs[c, r] * handle_vars[c, r] for c in range(n_call_centers) for r in range(n_regions)),
            "minimize"
        )

        # Constraints: Each region's call demand is met by the call centers
        for r in range(n_regions):
            model.addCons(quicksum(handle_vars[c, r] for c in range(n_call_centers)) == region_demand[r], f"Region_{r}_Demand")

        # Constraints: Only operational call centers can handle calls, and handle within limits
        for c in range(n_call_centers):
            for r in range(n_regions):
                model.addCons(handle_vars[c, r] <= operational_limits[c] * open_vars[c], f"Center_{c}_Operational_{r}")

        # Constraints: Call Centers cannot exceed their call handling capacities and must adhere to time windows
        for c in range(n_call_centers):
            model.addCons(quicksum(handle_vars[c, r] for r in range(n_regions)) <= capacities[c], f"Center_{c}_Capacity")
            model.addCons(quicksum(serve_vars[c, r] for r in range(n_regions)) <= time_windows[c] * open_vars[c], f"Center_{c}_TimeWindow")

        # Logical constraint: Ensure each open call center meets minimum service requirements
        for r in range(n_regions):
            for c in range(n_call_centers):
                model.addCons(serve_vars[c, r] <= open_vars[c], f"Serve_Open_Constraint_{c}_{r}")
            model.addCons(quicksum(serve_vars[c, r] for c in range(n_call_centers)) >= 1, f"Region_{r}_Service")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 85
    parameters = {
        'n_call_centers': 50,
        'n_regions': 160,
        'min_contact_cost': 10,
        'max_contact_cost': 250,
        'min_center_cost': 1000,
        'max_center_cost': 5000,
        'min_capacity': 1350,
        'max_capacity': 3000,
        'min_operational_limit': 200,
        'max_operational_limit': 750,
        'min_time_window': 80,
        'max_time_window': 168,
    }

    call_center_optimizer = CallCenterOptimization(parameters, seed=seed)
    instance = call_center_optimizer.generate_instance()
    solve_status, solve_time, objective_value = call_center_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")