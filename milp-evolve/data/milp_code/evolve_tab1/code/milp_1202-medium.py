import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HLASP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_hospitals_and_patients(self):
        hospitals = range(self.n_hospitals)
        patients = range(self.n_patients)

        # Appointment costs
        appointment_costs = np.random.randint(self.min_appointment_cost, self.max_appointment_cost + 1, (self.n_hospitals, self.n_patients))

        # Hospital costs and capacities
        hospital_costs = np.random.randint(self.min_hospital_cost, self.max_hospital_cost + 1, self.n_hospitals)
        hospital_capacities = np.random.randint(self.min_hospital_capacity, self.max_hospital_capacity + 1, self.n_hospitals)

        # Patient demands
        patient_demands = np.random.randint(self.min_patient_demand, self.max_patient_demand + 1, self.n_patients)

        res = {
            'hospitals': hospitals, 
            'patients': patients, 
            'appointment_costs': appointment_costs,
            'hospital_costs': hospital_costs,
            'hospital_capacities': hospital_capacities,
            'patient_demands': patient_demands,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        hospitals = instance['hospitals']
        patients = instance['patients']
        appointment_costs = instance['appointment_costs']
        hospital_costs = instance['hospital_costs']
        hospital_capacities = instance['hospital_capacities']
        patient_demands = instance['patient_demands']

        model = Model("HLASP")
        
        # Variables
        appointment_vars = { (h, p): model.addVar(vtype="B", name=f"appointment_{h+1}_{p+1}") for h in hospitals for p in patients}
        hospital_vars = { h: model.addVar(vtype="B", name=f"hospital_{h+1}") for h in hospitals }
        
        # Objective
        objective_expr = quicksum(appointment_costs[h, p] * appointment_vars[h, p] for h in hospitals for p in patients)
        objective_expr += quicksum(hospital_costs[h] * hospital_vars[h] for h in hospitals)
        
        model.setObjective(objective_expr, "minimize")
        
        # Constraints
        for p in patients:
            model.addCons(quicksum(appointment_vars[h, p] for h in hospitals) == 1, f"appointment_satisfy_{p+1}")

        M = sum(patient_demands)
        
        for h in hospitals:
            model.addCons(quicksum(appointment_vars[h, p] * patient_demands[p] for p in patients) <= hospital_capacities[h] * hospital_vars[h], f"capacity_{h+1}")
            model.addCons(quicksum(appointment_vars[h, p] for p in patients) <= M * hospital_vars[h], f"big_M_{h+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hospitals': 80,
        'n_patients': 100,
        'min_hospital_cost': 10000,
        'max_hospital_cost': 50000,
        'min_appointment_cost': 350,
        'max_appointment_cost': 1500,
        'min_hospital_capacity': 300,
        'max_hospital_capacity': 1500,
        'min_patient_demand': 8,
        'max_patient_demand': 20,
    }

    hl_asp = HLASP(parameters, seed=seed)
    instance = hl_asp.generate_hospitals_and_patients()
    solve_status, solve_time = hl_asp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")