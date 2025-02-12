import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CollegeHousingAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_colleges > 0 and self.n_houses > 0
        assert self.min_college_cost >= 0 and self.max_college_cost >= self.min_college_cost
        assert self.min_house_cost >= 0 and self.max_house_cost >= self.min_house_cost
        assert self.min_college_cap > 0 and self.max_college_cap >= self.min_college_cap

        college_costs = np.random.randint(self.min_college_cost, self.max_college_cost + 1, self.n_colleges)
        house_costs = np.random.randint(self.min_house_cost, self.max_house_cost + 1, (self.n_colleges, self.n_houses))
        capacities = np.random.randint(self.min_college_cap, self.max_college_cap + 1, self.n_colleges)
        demands = np.random.randint(1, 10, self.n_houses)
        comfort_levels = np.random.uniform(self.min_comfort, self.max_comfort, self.n_houses)
        maintenance_costs = np.random.uniform(self.min_maintenance_cost, self.max_maintenance_cost, self.n_colleges)

        neighboring_zones = []
        for _ in range(self.n_neighboring_pairs):
            h1 = random.randint(0, self.n_houses - 1)
            h2 = random.randint(0, self.n_houses - 1)
            if h1 != h2:
                neighboring_zones.append((h1, h2))

        construction_costs = np.random.uniform(5, 15, (self.n_colleges, self.n_houses))
        noise_limits = np.random.uniform(50, 100, self.n_houses)
        noise_levels = np.random.uniform(3, 7, self.n_houses)

        return {
            "college_costs": college_costs,
            "house_costs": house_costs,
            "capacities": capacities,
            "demands": demands,
            "comfort_levels": comfort_levels,
            "maintenance_costs": maintenance_costs,
            "neighboring_zones": neighboring_zones,
            "construction_costs": construction_costs,
            "noise_limits": noise_limits,
            "noise_levels": noise_levels
        }

    def solve(self, instance):
        college_costs = instance['college_costs']
        house_costs = instance['house_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        comfort_levels = instance['comfort_levels']
        maintenance_costs = instance['maintenance_costs']
        neighboring_zones = instance["neighboring_zones"]
        
        construction_costs = instance["construction_costs"]
        noise_limits = instance["noise_limits"]
        noise_levels = instance["noise_levels"]

        model = Model("CollegeHousingAllocation")
        n_colleges = len(college_costs)
        n_houses = len(house_costs[0])

        # Decision variables
        college_vars = {c: model.addVar(vtype="B", name=f"College_{c}") for c in range(n_colleges)}
        house_vars = {(c, h): model.addVar(vtype="B", name=f"College_{c}_House_{h}") for c in range(n_colleges) for h in range(n_houses)}
        maintenance_vars = {c: model.addVar(vtype="C", name=f"Maintenance_{c}") for c in range(n_colleges)}
        comfort_level_vars = {h: model.addVar(vtype="C", name=f"Comfort_{h}") for h in range(n_houses)}

        # New variables for added complexity
        construction_vars = {(c, h): model.addVar(vtype="B", name=f"Construction_{c}_{h}") for c in range(n_colleges) for h in range(n_houses)}
        noise_vars = {h: model.addVar(vtype="C", name=f"Noise_{h}") for h in range(n_houses)}

        # Objective: minimize the total cost including college costs, house costs, maintenance costs, and construction costs
        model.setObjective(
            quicksum(college_costs[c] * college_vars[c] for c in range(n_colleges)) +
            quicksum(house_costs[c, h] * house_vars[c, h] for c in range(n_colleges) for h in range(n_houses)) + 
            quicksum(maintenance_vars[c] for c in range(n_colleges)) +
            quicksum(comfort_level_vars[h] for h in range(n_houses)) +
            quicksum(construction_costs[c, h] * construction_vars[c, h] for c in range(n_colleges) for h in range(n_houses)),
            "minimize"
        )

        # Constraints: Each house demand is met by exactly one college
        for h in range(n_houses):
            model.addCons(quicksum(house_vars[c, h] for c in range(n_colleges)) == 1, f"House_{h}_Demand")
        
        # Constraints: Only open colleges can serve houses
        for c in range(n_colleges):
            for h in range(n_houses):
                model.addCons(house_vars[c, h] <= college_vars[c], f"College_{c}_Serve_{h}")
        
        # Constraints: Colleges cannot exceed their capacities
        for c in range(n_colleges):
            model.addCons(quicksum(demands[h] * house_vars[c, h] for h in range(n_houses)) <= capacities[c], f"College_{c}_Capacity")
        
        # Maintenance Cost Constraints
        for c in range(n_colleges):
            model.addCons(maintenance_vars[c] == self.maintenance_multiplier * college_vars[c], f"MaintenanceCost_{c}")

        # Comfort Level Constraints
        for h in range(n_houses):
            model.addCons(comfort_level_vars[h] == comfort_levels[h] * quicksum(house_vars[c, h] for c in range(n_colleges)), f"Comfort_{h}")

        # Neighboring Zone Noise Constraints
        for (h1, h2) in neighboring_zones:
            model.addCons(noise_vars[h1] + noise_vars[h2] <= noise_limits[h1], f"Noise_{h1}_{h2}")

        # New construction cost constraints
        for c in range(n_colleges):
            for h in range(n_houses):
                model.addCons(construction_vars[c, h] <= house_vars[c, h], f"Construction_{c}_{h}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_colleges': 200,
        'n_houses': 45,
        'min_house_cost': 1875,
        'max_house_cost': 5000,
        'min_college_cost': 5000,
        'max_college_cost': 15000,
        'min_college_cap': 75,
        'max_college_cap': 1050,
        'min_maintenance_cost': 3000,
        'max_maintenance_cost': 5000,
        'maintenance_multiplier': 1500.0,
        'n_neighboring_pairs': 300,
        'min_comfort': 0.59,
        'max_comfort': 9.0,
    }

    college_housing_optimizer = CollegeHousingAllocation(parameters, seed=seed)
    instance = college_housing_optimizer.generate_instance()
    solve_status, solve_time, objective_value = college_housing_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")