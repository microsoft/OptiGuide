import random
import time
import numpy as np
from pyscipopt import Model, quicksum


class EmergencyShelterResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def normal_int(self, size, mean, std_dev, lower_bound, upper_bound):
        return np.clip(np.round(np.random.normal(mean, std_dev, size)), lower_bound, upper_bound).astype(int)

    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers_emergency, 1) - rand(1, self.n_shelters))**2 +
            (rand(self.n_customers_emergency, 1) - rand(1, self.n_shelters))**2
        )
        return costs

    def generate_instance(self):
        demands = self.normal_int(self.n_customers_emergency, self.demand_mean, self.demand_std, self.demand_lower, self.demand_upper)
        capacities = self.normal_int(self.n_shelters, self.capacity_mean, self.capacity_std, self.capacity_lower, self.capacity_upper)
        fixed_costs = (
            self.normal_int(self.n_shelters, self.helipad_construction_cost_mean, self.helipad_construction_cost_std, self.helipad_construction_cost_lower, self.helipad_construction_cost_upper) * np.sqrt(capacities) +
            self.normal_int(self.n_shelters, self.medical_equipment_cost_mean, self.medical_equipment_cost_std, self.medical_equipment_cost_lower, self.medical_equipment_cost_upper)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        ideal_locations = np.random.choice(range(self.n_shelters), self.n_ideal_locations, replace=False)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'ideal_locations': ideal_locations
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        ideal_locations = instance['ideal_locations']
        
        n_customers_emergency = len(demands)
        n_shelters = len(capacities)
        
        model = Model("EmergencyShelterResourceAllocation")
        
        # Decision variables
        open_shelters = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_shelters)}
        serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers_emergency) for j in range(n_shelters)}
        unmet_demand_penalty = {i: model.addVar(vtype="C", name=f"Unmet_Demand_Penalty_{i}") for i in range(n_customers_emergency)}
        non_ideal_location_penalty = {j: model.addVar(vtype="C", name=f"Non_Ideal_Location_Penalty_{j}") for j in range(n_shelters)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_shelters[j] for j in range(n_shelters)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers_emergency) for j in range(n_shelters)) + \
                         quicksum(self.big_m_value * unmet_demand_penalty[i] for i in range(n_customers_emergency)) + \
                         quicksum(self.big_m_value * non_ideal_location_penalty[j] for j in range(n_shelters))

        model.setObjective(objective_expr, "minimize")

        # Constraints: each depraved area must be served by at least one shelter or get a penalty
        for i in range(n_customers_emergency):
            model.addCons(quicksum(serve[i, j] for j in range(n_shelters)) + unmet_demand_penalty[i] >= 1, f"Serving_Depraved_Areas_{i}")

        # Constraints: capacity limits at each shelter
        for j in range(n_shelters):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers_emergency)) <= capacities[j] * open_shelters[j], f"Shelter_Capacity_{j}")

        # General constraint on the total number of shelters to be opened
        model.addCons(quicksum(open_shelters[j] for j in range(n_shelters)) <= self.shelter_limit, "Number_of_Shelters")

        # Penalty for non-ideal shelter locations
        ideal_locations_set = set(ideal_locations)
        for j in range(n_shelters):
            if j not in ideal_locations_set:
                model.addCons(non_ideal_location_penalty[j] >= open_shelters[j], f"Non_Ideal_Location_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers_emergency': 400,
        'n_shelters': 50,
        'demand_mean': 1200,
        'demand_std': 140,
        'demand_lower': 60,
        'demand_upper': 1080,
        'capacity_mean': 2500,
        'capacity_std': 100,
        'capacity_lower': 28,
        'capacity_upper': 120,
        'helipad_construction_cost_mean': 1800,
        'helipad_construction_cost_std': 1800,
        'helipad_construction_cost_lower': 2700,
        'helipad_construction_cost_upper': 2700,
        'medical_equipment_cost_mean': 1575,
        'medical_equipment_cost_std': 75,
        'medical_equipment_cost_lower': 450,
        'medical_equipment_cost_upper': 3000,
        'ratio': 100.0,
        'continuous_assignment': 35,
        'shelter_limit': 1400,
        'big_m_value': 10000,
        'n_ideal_locations': 10
    }

    resource_allocation = EmergencyShelterResourceAllocation(parameters, seed=seed)
    instance = resource_allocation.generate_instance()
    solve_status, solve_time = resource_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")