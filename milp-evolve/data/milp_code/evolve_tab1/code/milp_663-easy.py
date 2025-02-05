import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HealthcareResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_hospitals > 0 and self.n_patients > 0

        patient_needs = np.random.randint(self.min_needs, self.max_needs, (self.n_patients, self.n_resources))
        hospital_capacities = np.random.randint(self.min_capacity, self.max_capacity, (self.n_hospitals, self.n_resources))
        
        # Creating hospital-patient assignment preferences
        hospital_preferences = np.random.rand(self.n_hospitals, self.n_patients)
        
        return {
            "patient_needs": patient_needs,
            "hospital_capacities": hospital_capacities,
            "hospital_preferences": hospital_preferences
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        patient_needs = instance['patient_needs']
        hospital_capacities = instance['hospital_capacities']
        hospital_preferences = instance['hospital_preferences']

        model = Model("HealthcareResourceAllocation")

        # Decision variables
        alloc_vars = {(h, p): model.addVar(vtype="B", name=f"Alloc_{h}_{p}") for h in range(self.n_hospitals) for p in range(self.n_patients)}
        
        # Objective: maximize the total preference score
        objective_expr = quicksum(hospital_preferences[h, p] * alloc_vars[h, p] for h in range(self.n_hospitals) for p in range(self.n_patients))
        
        # Constraints: Each patient's needs must be met if allocated
        for p in range(self.n_patients):
            for r in range(self.n_resources):
                model.addCons(quicksum(patient_needs[p, r] * alloc_vars[h, p] for h in range(self.n_hospitals)) >= patient_needs[p, r], f"Patient_{p}_Need_{r}")
        
        # Constraints: Resource capacities of each hospital should not be exceeded
        for h in range(self.n_hospitals):
            for r in range(self.n_resources):
                model.addCons(quicksum(patient_needs[p, r] * alloc_vars[h, p] for p in range(self.n_patients)) <= hospital_capacities[h, r], f"Hospital_{h}_Capacity_{r}")

        model.setObjective(objective_expr, "maximize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hospitals': 20,
        'n_patients': 25,
        'n_resources': 2,
        'min_needs': 1,
        'max_needs': 70,
        'min_capacity': 150,
        'max_capacity': 200,
    }

    healthcare = HealthcareResourceAllocation(parameters, seed=42)
    instance = healthcare.generate_instance()
    solve_status, solve_time = healthcare.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")