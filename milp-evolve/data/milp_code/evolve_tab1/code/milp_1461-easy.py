import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class UrbanRedevelopmentOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_zones > 0 and self.n_trucks > 0
        assert self.min_dispatch_cost >= 0 and self.max_dispatch_cost >= self.min_dispatch_cost
        assert self.min_handling_cost >= 0 and self.max_handling_cost >= self.min_handling_cost
        assert self.min_prep_cost >= 0 and self.max_prep_cost >= self.min_prep_cost

        dispatch_costs = np.random.randint(self.min_dispatch_cost, self.max_dispatch_cost + 1, self.n_trucks)
        handling_costs = np.random.randint(self.min_handling_cost, self.max_handling_cost + 1, (self.n_trucks, self.n_zones))
        preparation_costs = np.random.randint(self.min_prep_cost, self.max_prep_cost + 1, self.n_trucks)
        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_zones).astype(int)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_trucks)

        # Generate additional data for green spaces, public art, and economic scores
        green_space_cost = np.random.randint(100, 200, self.n_zones)
        public_art_cost = np.random.randint(150, 250, self.n_zones)
        economic_score = np.random.uniform(0.5, 1.5, self.n_zones)  # Represents economic equity score

        return {
            "dispatch_costs": dispatch_costs,
            "handling_costs": handling_costs,
            "preparation_costs": preparation_costs,
            "demands": demands,
            "capacities": capacities,
            "green_space_cost": green_space_cost,
            "public_art_cost": public_art_cost,
            "economic_score": economic_score
        }

    # MILP modeling
    def solve(self, instance):
        dispatch_costs = instance["dispatch_costs"]
        handling_costs = instance["handling_costs"]
        preparation_costs = instance["preparation_costs"]
        demands = instance["demands"]
        capacities = instance["capacities"]
        green_space_cost = instance["green_space_cost"]
        public_art_cost = instance["public_art_cost"]
        economic_score = instance["economic_score"]

        model = Model("UrbanRedevelopmentOptimization")
        n_trucks = len(capacities)
        n_zones = len(demands)

        MaxLimit = self.max_limit

        # Decision variables
        number_trucks = {t: model.addVar(vtype="B", name=f"Truck_{t}") for t in range(n_trucks)}
        number_packages = {(t, z): model.addVar(vtype="B", name=f"Truck_{t}_Zone_{z}") for t in range(n_trucks) for z in range(n_zones)}
        meal_prep_vars = {t: model.addVar(vtype="B", name=f"Truck_{t}_Meal_Prep") for t in range(n_trucks)}
        green_space_zone = {z: model.addVar(vtype="B", name=f"Green_Space_Zone_{z}") for z in range(n_zones)}
        public_art_zone = {z: model.addVar(vtype="B", name=f"Public_Art_Zone_{z}") for z in range(n_zones)}
        economic_score_zone = {z: model.addVar(vtype="C", name=f"Economic_Score_Zone_{z}") for z in range(n_zones)}

        # Objective function
        model.setObjective(
            quicksum(dispatch_costs[t] * number_trucks[t] for t in range(n_trucks)) +
            quicksum(handling_costs[t, z] * number_packages[t, z] for t in range(n_trucks) for z in range(n_zones)) +
            quicksum(preparation_costs[t] * meal_prep_vars[t] for t in range(n_trucks)) +
            quicksum(green_space_cost[z] * green_space_zone[z] for z in range(n_zones)) +
            quicksum(public_art_cost[z] * public_art_zone[z] for z in range(n_zones)) +
            quicksum(economic_score[z] * economic_score_zone[z] for z in range(n_zones)),
            "minimize"
        )

        # Constraints: Each zone must receive deliveries
        for z in range(n_zones):
            model.addCons(quicksum(number_packages[t, z] for t in range(n_trucks)) >= 1, f"Zone_{z}_Demand")

        # Constraints: Each truck cannot exceed its capacity
        for t in range(n_trucks):
            model.addCons(quicksum(demands[z] * number_packages[t, z] for z in range(n_zones)) <= capacities[t], f"Truck_{t}_Capacity")

        # Constraints: Only dispatched trucks can make deliveries
        for t in range(n_trucks):
            for z in range(n_zones):
                model.addCons(number_packages[t, z] <= number_trucks[t], f"Truck_{t}_Deliver_{z}")

        # New Constraints: Only trucks with meal preparation can make deliveries
        for t in range(n_trucks):
            for z in range(n_zones):
                model.addCons(number_packages[t, z] <= meal_prep_vars[t], f"Truck_{t}_Prep_{z}")

        # Ensure each zone has a minimum number of green spaces
        for z in range(n_zones):
            model.addCons(green_space_zone[z] >= self.min_green_spaces, f"Zone_{z}_Green_Spaces")

        # Ensure each zone has a minimum number of public art installations
        for z in range(n_zones):
            model.addCons(public_art_zone[z] >= self.min_public_art, f"Zone_{z}_Public_Art")

        # Economic equity constraints
        for z in range(n_zones):
            model.addCons(economic_score_zone[z] >= 1, f"Zone_{z}_Economic_Score")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == "optimal":
            objective_value = model.getObjVal()
        else:
            objective_value = None

        return model.getStatus(), end_time - start_time, objective_value

if __name__ == "__main__":
    seed = 42
    parameters = {
        'n_trucks': 50,
        'n_zones': 75,
        'min_dispatch_cost': 9,
        'max_dispatch_cost': 750,
        'min_handling_cost': 56,
        'max_handling_cost': 702,
        'mean_demand': 8,
        'std_dev_demand': 25,
        'min_capacity': 13,
        'max_capacity': 3000,
        'max_limit': 562,
        'min_prep_cost': 100,
        'max_prep_cost': 1050,
        'min_green_spaces': 1,
        'min_public_art': 1,
    }

    urban_optimizer = UrbanRedevelopmentOptimization(parameters, seed)
    instance = urban_optimizer.generate_instance()
    solve_status, solve_time, objective_value = urban_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")