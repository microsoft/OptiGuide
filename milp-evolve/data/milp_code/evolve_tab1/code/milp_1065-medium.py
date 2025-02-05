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
        
        # New components for more complexity
        water_availability = np.random.randint(1, 100, self.n_fields)
        crop_water_needs = np.random.randint(1, 10, self.n_crops)
        environmental_impact = np.random.uniform(0.1, 1.0, self.n_crops)
        profit_per_crop = np.random.uniform(100, 10000, self.n_crops)

        # Generate adjacency matrix for logical constraints
        adjacency_matrix = np.random.randint(0, 2, (self.n_fields, self.n_fields))  # Binary adjacency matrix
        
        return {
            "field_costs": field_costs,
            "fertilizer_costs": fertilizer_costs,
            "nutrient_capacity": nutrient_capacity,
            "crop_nutrient_needs": crop_nutrient_needs,
            "adjacency_matrix": adjacency_matrix,
            "water_availability": water_availability,
            "crop_water_needs": crop_water_needs,
            "environmental_impact": environmental_impact,
            "profit_per_crop": profit_per_crop
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        field_costs = instance['field_costs']
        fertilizer_costs = instance['fertilizer_costs']
        nutrient_capacity = instance['nutrient_capacity']
        crop_nutrient_needs = instance['crop_nutrient_needs']
        adjacency_matrix = instance['adjacency_matrix']
        water_availability = instance['water_availability']
        crop_water_needs = instance['crop_water_needs']
        environmental_impact = instance['environmental_impact']
        profit_per_crop = instance['profit_per_crop']
        
        model = Model("SustainableFarmingOptimization")
        n_fields = len(field_costs)
        n_crops = len(fertilizer_costs[0])
        
        # Decision variables
        field_vars = {f: model.addVar(vtype="B", name=f"Field_{f}") for f in range(n_fields)}
        serve_vars = {(f, c): model.addVar(vtype="B", name=f"Field_{f}_Crop_{c}") for f in range(n_fields) for c in range(n_crops)}
        
        # Auxiliary Variables for Logical Constraints
        pairwise_field_vars = {(f1, f2): model.addVar(vtype="B", name=f"Field_{f1}_Adjacent_{f2}") for f1 in range(n_fields) for f2 in range(n_fields) if adjacency_matrix[f1, f2] == 1}
        
        # Objective: minimize the total cost and maximize profit (field + fertilizer + environmental impact - profit of crops)
        model.setObjective(
            quicksum(field_costs[f] * field_vars[f] for f in range(n_fields)) +
            quicksum(fertilizer_costs[f, c] * serve_vars[f, c] for f in range(n_fields) for c in range(n_crops)) +
            quicksum(environmental_impact[c] * serve_vars[f, c] for f in range(n_fields) for c in range(n_crops)) -
            quicksum(profit_per_crop[c] * serve_vars[f, c] for f in range(n_fields) for c in range(n_crops)),
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
        
        # New Water Constraints: Each crop water need is met and doesn't exceed field's availability
        for f in range(n_fields):
            model.addCons(quicksum(crop_water_needs[c] * serve_vars[f, c] for c in range(n_crops)) <= water_availability[f], f"Field_{f}_Water")

        # Logical Constraints: Ensure adjacent fields are either both open or both closed for specific crops
        for (f1, f2) in pairwise_field_vars:
            model.addCons(pairwise_field_vars[f1, f2] <= quicksum(field_vars[f] for f in [f1, f2])) # If one of the fields is not used, then the pairwise variable is 0
            model.addCons(quicksum(field_vars[f] for f in [f1, f2]) <= 2 * pairwise_field_vars[f1, f2]) # If both of the fields are used, then the pairwise variable is 1

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_fields': 81,
        'n_crops': 58,
        'min_fertilizer_cost': 1800,
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