import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexHLASP:
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

        # Travel costs (distances)
        travel_costs = np.random.randint(self.min_travel_cost, self.max_travel_cost + 1, (self.n_hospitals, self.n_patients))

        # Hospital costs and capacities
        hospital_costs = np.random.randint(self.min_hospital_cost, self.max_hospital_cost + 1, self.n_hospitals)
        hospital_capacities = np.random.randint(self.min_hospital_capacity, self.max_hospital_capacity + 1, self.n_hospitals)

        # Patient demands
        patient_demands = np.random.randint(self.min_patient_demand, self.max_patient_demand + 1, self.n_patients)

        # Patient priority scores
        patient_priorities = np.random.randint(self.min_patient_priority, self.max_patient_priority + 1, self.n_patients)

        res = {
            'hospitals': hospitals, 
            'patients': patients, 
            'appointment_costs': appointment_costs,
            'travel_costs': travel_costs,
            'hospital_costs': hospital_costs,
            'hospital_capacities': hospital_capacities,
            'patient_demands': patient_demands,
            'patient_priorities': patient_priorities,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        hospitals = instance['hospitals']
        patients = instance['patients']
        appointment_costs = instance['appointment_costs']
        travel_costs = instance['travel_costs']
        hospital_costs = instance['hospital_costs']
        hospital_capacities = instance['hospital_capacities']
        patient_demands = instance['patient_demands']
        patient_priorities = instance['patient_priorities']

        model = Model("ComplexHLASP")
        
        # Variables
        appointment_vars = { (h, p): model.addVar(vtype="B", name=f"appointment_{h+1}_{p+1}") for h in hospitals for p in patients}
        hospital_vars = { h: model.addVar(vtype="B", name=f"hospital_{h+1}") for h in hospitals }
        
        # Objective
        # Minimize total cost considering appointment costs, hospital costs, and travel costs
        objective_expr = quicksum(appointment_costs[h, p] * appointment_vars[h, p] for h in hospitals for p in patients)
        objective_expr += quicksum(travel_costs[h, p] * appointment_vars[h, p] for h in hospitals for p in patients)
        objective_expr += quicksum(hospital_costs[h] * hospital_vars[h] for h in hospitals)
        
        model.setObjective(objective_expr, "minimize")

        # Constraints
        # Ensure each patient is appointed exactly once at one hospital.
        for p in patients:
            model.addCons(quicksum(appointment_vars[h, p] for h in hospitals) == 1, f"appointment_satisfy_{p+1}")

        # Capacity constraints for hospitals.
        for h in hospitals:
            model.addCons(quicksum(appointment_vars[h, p] * patient_demands[p] for p in patients) <= hospital_capacities[h], f"capacity_{h+1}")

        # Maximum working hours for hospitals, considering each hospital can only operate for a limited duration.
        max_hospital_hours = self.max_hospital_hours
        per_patient_time = self.per_patient_time
        for h in hospitals:
            model.addCons(quicksum(appointment_vars[h, p] for p in patients) * per_patient_time <= max_hospital_hours * hospital_vars[h], f"working_hours_{h+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hospitals': 800,
        'n_patients': 10,
        'min_hospital_cost': 10000,
        'max_hospital_cost': 50000,
        'min_appointment_cost': 35,
        'max_appointment_cost': 75,
        'min_travel_cost': 7,
        'max_travel_cost': 10,
        'min_hospital_capacity': 30,
        'max_hospital_capacity': 75,
        'min_patient_demand': 6,
        'max_patient_demand': 10,
        'min_patient_priority': 0,
        'max_patient_priority': 25,
        'max_hospital_hours': 10000,
        'per_patient_time': 450,
    }

    hl_asp = ComplexHLASP(parameters, seed=seed)
    instance = hl_asp.generate_hospitals_and_patients()
    solve_status, solve_time = hl_asp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")