import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CourierServiceOptimization:
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

        return {
            "dispatch_costs": dispatch_costs,
            "handling_costs": handling_costs,
            "preparation_costs": preparation_costs,
            "demands": demands,
            "capacities": capacities
        }

    # MILP modeling
    def solve(self, instance):
        dispatch_costs = instance["dispatch_costs"]
        handling_costs = instance["handling_costs"]
        preparation_costs = instance["preparation_costs"]
        demands = instance["demands"]
        capacities = instance["capacities"]

        model = Model("CourierServiceOptimization")
        n_trucks = len(capacities)
        n_zones = len(demands)

        MaxLimit = self.max_limit

        # Decision variables
        number_trucks = {t: model.addVar(vtype="B", name=f"Truck_{t}") for t in range(n_trucks)}
        number_packages = {(t, z): model.addVar(vtype="B", name=f"Truck_{t}_Zone_{z}") for t in range(n_trucks) for z in range(n_zones)}
        meal_prep_vars = {t: model.addVar(vtype="B", name=f"Truck_{t}_Meal_Prep") for t in range(n_trucks)}

        # Objective function
        model.setObjective(
            quicksum(dispatch_costs[t] * number_trucks[t] for t in range(n_trucks)) +
            quicksum(handling_costs[t, z] * number_packages[t, z] for t in range(n_trucks) for z in range(n_zones)) +
            quicksum(preparation_costs[t] * meal_prep_vars[t] for t in range(n_trucks)),
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
        'min_dispatch_cost': 187,
        'max_dispatch_cost': 750,
        'min_handling_cost': 225,
        'max_handling_cost': 937,
        'mean_demand': 35,
        'std_dev_demand': 50,
        'min_capacity': 52,
        'max_capacity': 3000,
        'max_limit': 750,
        'min_prep_cost': 1000,
        'max_prep_cost': 1050,
    }

    courier_optimizer = CourierServiceOptimization(parameters, seed)
    instance = courier_optimizer.generate_instance()
    solve_status, solve_time, objective_value = courier_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")