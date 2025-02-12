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
        
        adjacency_matrix = np.random.randint(0, 2, (self.n_fields, self.n_fields))  # Binary adjacency matrix
        
        parking_capacity = np.random.randint(1, self.max_parking_capacity, size=self.n_parking_zones)
        parking_zones = {i: np.random.choice(range(self.n_fields), size=self.n_parking_in_zone, replace=False) for i in range(self.n_parking_zones)}
        time_windows = {i: (np.random.randint(0, self.latest_delivery_time // 2),
                            np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) for i in range(self.n_fields)}
        uncertainty = {i: np.random.normal(0, self.time_uncertainty_stddev, size=2) for i in range(self.n_fields)}
        eco_friendly_zones = np.random.choice(range(self.n_fields), size=self.n_eco_friendly_zones, replace=False)
        co2_saving = {i: np.random.uniform(0, self.max_co2_saving) for i in eco_friendly_zones}

        return {
            "field_costs": field_costs,
            "fertilizer_costs": fertilizer_costs,
            "nutrient_capacity": nutrient_capacity,
            "crop_nutrient_needs": crop_nutrient_needs,
            "adjacency_matrix": adjacency_matrix,
            "parking_capacity": parking_capacity,
            "parking_zones": parking_zones,
            "time_windows": time_windows,
            "uncertainty": uncertainty,
            "eco_friendly_zones": eco_friendly_zones,
            "co2_saving": co2_saving,
            "sustainability_constraint": np.random.uniform(0, self.min_sustainability_requirement)
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        field_costs = instance['field_costs']
        fertilizer_costs = instance['fertilizer_costs']
        nutrient_capacity = instance['nutrient_capacity']
        crop_nutrient_needs = instance['crop_nutrient_needs']
        adjacency_matrix = instance['adjacency_matrix']
        parking_capacity = instance['parking_capacity']
        parking_zones = instance['parking_zones']
        time_windows = instance['time_windows']
        uncertainty = instance['uncertainty']
        eco_friendly_zones = instance['eco_friendly_zones']
        co2_saving = instance['co2_saving']
        sustainability_constraint = instance['sustainability_constraint']

        model = Model("SustainableFarmingOptimization")
        n_fields = len(field_costs)
        n_crops = len(fertilizer_costs[0])
        
        # Decision variables
        field_vars = {f: model.addVar(vtype="B", name=f"Field_{f}") for f in range(n_fields)}
        serve_vars = {(f, c): model.addVar(vtype="B", name=f"Field_{f}_Crop_{c}") for f in range(n_fields) for c in range(n_crops)}
        
        # Auxiliary Variables for Logical Constraints
        pairwise_field_vars = {(f1, f2): model.addVar(vtype="B", name=f"Field_{f1}_Adjacent_{f2}") for f1 in range(n_fields) for f2 in range(n_fields) if adjacency_matrix[f1, f2] == 1}
        
        # Additional variables for new constraints
        time_vars = {f: model.addVar(vtype='C', name=f"t_{f}") for f in range(n_fields)}
        early_penalty_vars = {f: model.addVar(vtype='C', name=f"e_{f}") for f in range(n_fields)}
        late_penalty_vars = {f: model.addVar(vtype='C', name=f"l_{f}") for f in range(n_fields)}
        parking_vars = {col: model.addVar(vtype="B", name=f"p_{col}") for zone, cols in parking_zones.items() for col in cols}
        
        # Objective: minimize the total cost (field + fertilizer + penalties)
        model.setObjective(
            quicksum(field_costs[f] * field_vars[f] for f in range(n_fields)) +
            quicksum(fertilizer_costs[f, c] * serve_vars[f, c] for f in range(n_fields) for c in range(n_crops)) +
            quicksum(early_penalty_vars[f] + late_penalty_vars[f] for f in range(n_fields)), 
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
        
        # Logical Constraints: Ensure adjacent fields are either both open or both closed for specific crops
        for (f1, f2) in pairwise_field_vars:
            model.addCons(pairwise_field_vars[f1, f2] <= quicksum(field_vars[f] for f in [f1, f2])) # If one of the fields is not used, then the pairwise variable is 0
            model.addCons(quicksum(field_vars[f] for f in [f1, f2]) <= 2 * pairwise_field_vars[f1, f2]) # If both of the fields are used, then the pairwise variable is 1
        
        # Delivery Time Windows with Uncertainty
        for f in range(n_fields):
            start_window, end_window = time_windows[f]
            uncertainty_start, uncertainty_end = uncertainty[f]
            model.addCons(time_vars[f] >= start_window + uncertainty_start, f"time_window_start_{f}")
            model.addCons(time_vars[f] <= end_window + uncertainty_end, f"time_window_end_{f}")
            model.addCons(early_penalty_vars[f] >= start_window + uncertainty_start - time_vars[f], f"early_penalty_{f}")
            model.addCons(late_penalty_vars[f] >= time_vars[f] - (end_window + uncertainty_end), f"late_penalty_{f}")

        # Parking constraints using Big M method
        M = self.big_M_constant
        for zone, cols in parking_zones.items():
            for col in cols:
                model.addCons(field_vars[col] <= parking_vars[col] * M, f"occupy_{col}_big_m")
                model.addCons(parking_vars[col] <= field_vars[col] + (1 - field_vars[col]) * M, f"occupy_{col}_reverse_big_m")

            model.addCons(quicksum(parking_vars[col] for col in cols) <= parking_capacity[zone], f"parking_limit_{zone}")

        # Sustainability constraints (CO2 saving)
        model.addCons(quicksum(co2_saving[f] * field_vars[f] for f in eco_friendly_zones) >= sustainability_constraint, "sustainability")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_fields': 108,
        'n_crops': 39,
        'min_fertilizer_cost': 45,
        'max_fertilizer_cost': 3000,
        'min_field_cost': 1875,
        'max_field_cost': 5000,
        'min_nutrient_capacity': 8,
        'max_nutrient_capacity': 600,
        'big_M_constant': 3000,
        'latest_delivery_time': 2160,
        'time_uncertainty_stddev': 2,
        'n_parking_zones': 30,
        'n_parking_in_zone': 7,
        'max_parking_capacity': 200,
        'n_eco_friendly_zones': 22,
        'max_co2_saving': 100,
        'min_sustainability_requirement': 25,
    }

    sustainable_farming_optimizer = SustainableFarmingOptimization(parameters, seed=42)
    instance = sustainable_farming_optimizer.generate_instance()
    solve_status, solve_time, objective_value = sustainable_farming_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")