import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SustainableFarmingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_fields > 0 and self.n_crops > 0
        assert self.min_fertilizer_cost >= 0 and self.max_fertilizer_cost >= self.min_fertilizer_cost
        assert self.min_field_cost >= 0 and self.max_field_cost >= self.min_field_cost
        assert self.min_nutrient_capacity > 0 and self.max_nutrient_capacity >= self.min_nutrient_capacity
        
        field_costs = np.random.randint(self.min_field_cost, self.max_field_cost + 1, self.n_fields)
        fertilizer_costs = np.random.randint(self.min_fertilizer_cost, self.max_fertilizer_cost + 1, (self.n_fields, self.n_crops))
        nutrient_capacity = np.random.randint(self.min_nutrient_capacity, self.max_nutrient_capacity + 1, self.n_fields)
        crop_nutrient_needs = np.random.randint(1, 10, self.n_crops)
        
        return {
            "field_costs": field_costs,
            "fertilizer_costs": fertilizer_costs,
            "nutrient_capacity": nutrient_capacity,
            "crop_nutrient_needs": crop_nutrient_needs
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        field_costs = instance['field_costs']
        fertilizer_costs = instance['fertilizer_costs']
        nutrient_capacity = instance['nutrient_capacity']
        crop_nutrient_needs = instance['crop_nutrient_needs']
        
        model = Model("SustainableFarmingOptimization")
        n_fields = len(field_costs)
        n_crops = len(fertilizer_costs[0])
        
        # Decision variables
        field_vars = {f: model.addVar(vtype="B", name=f"Field_{f}") for f in range(n_fields)}
        serve_vars = {(f, c): model.addVar(vtype="B", name=f"Field_{f}_Crop_{c}") for f in range(n_fields) for c in range(n_crops)}
        
        # Objective: minimize the total cost (field + fertilizer)
        model.setObjective(
            quicksum(field_costs[f] * field_vars[f] for f in range(n_fields)) +
            quicksum(fertilizer_costs[f, c] * serve_vars[f, c] for f in range(n_fields) for c in range(n_crops)), 
            "minimize"
        )
        
        # Constraints: Each crop nutrient need is met
        for c in range(n_crops):
            model.addCons(quicksum(serve_vars[f, c] for f in range(n_fields)) == 1, f"Crop_{c}_Nutrient")

        # Constraints: Only open fields can serve crops
        for f in range(n_fields):
            for c in range(n_crops):
                model.addCons(serve_vars[f, c] <= field_vars[f], f"Field_{f}_Serve_{c}")

        # Constraints: Fields cannot exceed their nutrient capacity
        for f in range(n_fields):
            model.addCons(quicksum(crop_nutrient_needs[c] * serve_vars[f, c] for c in range(n_crops)) <= nutrient_capacity[f], f"Field_{f}_Capacity")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_fields': 324,
        'n_crops': 39,
        'min_fertilizer_cost': 900,
        'max_fertilizer_cost': 3000,
        'min_field_cost': 1875,
        'max_field_cost': 5000,
        'min_nutrient_capacity': 165,
        'max_nutrient_capacity': 1200,
    }
    sustainable_farming_optimizer = SustainableFarmingOptimization(parameters, seed=42)
    instance = sustainable_farming_optimizer.generate_instance()
    solve_status, solve_time, objective_value = sustainable_farming_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")