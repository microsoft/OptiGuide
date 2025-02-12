import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HelicopterCargoAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_helicopters(self, num_helicopters):
        return [{'max_weight': np.random.randint(self.min_weight, self.max_weight),
                 'max_volume': np.random.uniform(self.min_volume, self.max_volume)}
                for _ in range(num_helicopters)]

    def generate_cargos(self, num_cargos):
        return [{'weight': np.random.randint(self.min_cargo_weight, self.max_cargo_weight),
                 'volume': np.random.uniform(self.min_cargo_volume, self.max_cargo_volume),
                 'value': np.random.randint(self.min_cargo_value, self.max_cargo_value)}
                for _ in range(num_cargos)]

    def generate_instances(self):
        num_helicopters = np.random.randint(self.min_num_helicopters, self.max_num_helicopters + 1)
        num_cargos = np.random.randint(self.min_num_cargos, self.max_num_cargos + 1)

        helicopters = self.generate_helicopters(num_helicopters)
        cargos = self.generate_cargos(num_cargos)

        res = {'helicopters': helicopters, 'cargos': cargos}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        helicopters = instance['helicopters']
        cargos = instance['cargos']

        num_helicopters = len(helicopters)
        num_cargos = len(cargos)

        model = Model("HelicopterCargoAllocation")
        
        # Create binary variables indicating whether each helicopter is used
        heli_vars = {i: model.addVar(vtype="B", name=f"Heli_{i}") for i in range(num_helicopters)}

        # Create binary variables indicating whether cargo is assigned to a helicopter
        hcargo_vars = {(i, j): model.addVar(vtype="B", name=f"HCargo_{i}_{j}") 
                       for i in range(num_helicopters) for j in range(num_cargos)}

        # Objective function - maximize the total value of the transported cargo
        objective_expr = quicksum(hcargo_vars[(i, j)] * cargos[j]['value']
                                  for i in range(num_helicopters) for j in range(num_cargos))

        # Constraints - weight and volume for each helicopter
        for i in range(num_helicopters):
            model.addCons(quicksum(hcargo_vars[(i, j)] * cargos[j]['weight'] for j in range(num_cargos)) <= helicopters[i]['max_weight'],
                          name=f"WeightCons_{i}")
            model.addCons(quicksum(hcargo_vars[(i, j)] * cargos[j]['volume'] for j in range(num_cargos)) <= helicopters[i]['max_volume'],
                          name=f"VolumeCons_{i}")

        # Each cargo can only be assigned to one helicopter
        for j in range(num_cargos):
            model.addCons(quicksum(hcargo_vars[(i, j)] for i in range(num_helicopters)) <= 1,
                          name=f"CargoAssignmentCons_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_num_helicopters': 6,
        'max_num_helicopters': 20,
        'min_num_cargos': 20,
        'max_num_cargos': 700,
        'min_weight': 1000,
        'max_weight': 5000,
        'min_volume': 10.0,
        'max_volume': 50.0,
        'min_cargo_weight': 700,
        'max_cargo_weight': 3000,
        'min_cargo_volume': 6.0,
        'max_cargo_volume': 100.0,
        'min_cargo_value': 1000,
        'max_cargo_value': 2000,
    }

    cargo_alloc = HelicopterCargoAllocation(parameters, seed=seed)
    instance = cargo_alloc.generate_instances()
    solve_status, solve_time = cargo_alloc.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")