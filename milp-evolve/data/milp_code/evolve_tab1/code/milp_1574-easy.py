import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EquipmentAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        utility = np.random.normal(loc=self.utility_mean, scale=self.utility_std, size=self.number_of_equipments).astype(int)
        weights = np.random.normal(loc=self.weight_mean, scale=self.weight_std, size=self.number_of_equipments).astype(int)
        volumes = np.random.normal(loc=self.volume_mean, scale=self.volume_std, size=self.number_of_equipments).astype(int)

        # Ensure non-negative values
        utility = np.clip(utility, self.min_utility, self.max_utility)
        weights = np.clip(weights, self.min_weight, self.max_weight)
        volumes = np.clip(volumes, self.min_volume, self.max_volume)

        volume_capacities = np.random.randint(
            0.4 * volumes.sum() // self.number_of_camps,
            0.6 * volumes.sum() // self.number_of_camps,
            size=self.number_of_camps
        )

        budget_capacities = np.random.randint(
            0.4 * weights.sum() // self.number_of_camps,
            0.6 * weights.sum() // self.number_of_camps,
            size=self.number_of_camps
        )

        res = {'utility': utility,
               'weights': weights,
               'volumes': volumes,
               'volume_capacities': volume_capacities,
               'budget_capacities': budget_capacities}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        utility = instance['utility']
        weights = instance['weights']
        volumes = instance['volumes']
        volume_capacities = instance['volume_capacities']
        budget_capacities = instance['budget_capacities']
        
        number_of_equipments = len(utility)
        number_of_camps = len(volume_capacities)
        
        model = Model("EquipmentAllocation")
        EquipmentVars = {}

        # Decision variables: x[i][j] = 1 if equipment i is allocated to camp j
        for i in range(number_of_equipments):
            for j in range(number_of_camps):
                EquipmentVars[(i, j)] = model.addVar(vtype="B", name=f"EquipmentVars_{i}_{j}")

        # Objective: Maximize total utility
        objective_expr = quicksum(utility[i] * EquipmentVars[(i, j)] for i in range(number_of_equipments) for j in range(number_of_camps))

        # Constraints: Each equipment can be in one camp at most
        for i in range(number_of_equipments):
            model.addCons(
                quicksum(EquipmentVars[(i, j)] for j in range(number_of_camps)) <= 1,
                f"EquipmentAllocation_{i}"
            )

        # Constraints: Total volume in each camp must not exceed its capacity
        for j in range(number_of_camps):
            model.addCons(
                quicksum(volumes[i] * EquipmentVars[(i, j)] for i in range(number_of_equipments)) <= volume_capacities[j],
                f"VolumeConstraints_{j}"
            )

        # Constraints: Total weight in each camp must not exceed the budget capacity
        for j in range(number_of_camps):
            model.addCons(
                quicksum(weights[i] * EquipmentVars[(i, j)] for i in range(number_of_equipments)) <= budget_capacities[j],
                f"BudgetConstraints_{j}"
            )
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_equipments': 500,
        'number_of_camps': 40,
        'min_utility': 0,
        'max_utility': 750,
        'utility_mean': 40,
        'utility_std': 50,
        'weight_mean': 60,
        'weight_std': 50,
        'min_weight': 0,
        'max_weight': 50,
        'volume_mean': 75,
        'volume_std': 1,
        'min_volume': 0,
        'max_volume': 80,
    }

    allocation = EquipmentAllocation(parameters, seed=seed)
    instance = allocation.generate_instance()
    solve_status, solve_time = allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")