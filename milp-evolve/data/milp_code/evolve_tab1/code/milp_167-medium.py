import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MedicalResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_facilities(self, n_facilities, n_resources, n_staff):
        facilities = {
            'capacity_resources': np.random.randint(5, 20, size=(n_facilities, n_resources)),
            'capacity_staff': np.random.randint(5, 20, size=(n_facilities, n_staff))
        }
        return facilities

    def generate_patient_data(self, n_patients, n_facilities):
        patient_data = {
            'resource_demand': np.random.randint(1, 5, size=(n_patients, n_facilities)),
            'staff_demand': np.random.randint(1, 5, size=(n_patients, n_facilities)),
            'travel_time': np.random.randint(10, 60, size=(n_patients, n_facilities)), # Simulated travel times
            'treatment_time': np.random.randint(5, 30, size=n_patients) # Treatment duration for each patient
        }
        return patient_data

    def generate_instances(self):
        n_facilities = np.random.randint(self.min_facilities, self.max_facilities + 1)
        n_patients = np.random.randint(self.min_patients, self.max_patients + 1)
        
        facilities = self.generate_facilities(n_facilities, self.n_resources, self.n_staff)
        patients = self.generate_patient_data(n_patients, n_facilities)

        res = {'facilities': facilities, 'patients': patients, 'n_facilities': n_facilities, 'n_patients': n_patients}
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        facilities = instance['facilities']
        patients = instance['patients']
        n_facilities = instance['n_facilities']
        n_patients = instance['n_patients']

        model = Model("MedicalResourceAllocation")

        # Variables to represent resource, staff allocation and travel assignments
        alloc_resource = {}  # Binary variable: 1 if resource is allocated, 0 otherwise
        alloc_staff = {}  # Binary variable: 1 if staff is allocated, 0 otherwise
        travel = {}  # Binary variable: 1 if travel is assigned, 0 otherwise

        for p in range(n_patients):
            for f in range(n_facilities):
                alloc_resource[p, f] = model.addVar(vtype="B", name=f"alloc_resource_{p}_{f}")
                alloc_staff[p, f] = model.addVar(vtype="B", name=f"alloc_staff_{p}_{f}")
                travel[p, f] = model.addVar(vtype="B", name=f"travel_{p}_{f}")

        ############### Objective Function ################
        # Minimize the total travel time and ensure adequate resource allocation
        travel_time = quicksum(patients['travel_time'][p, f] * travel[p, f] for p in range(n_patients) for f in range(n_facilities))
        model.setObjective(travel_time, "minimize")

        ############### Constraints ###################
        for p in range(n_patients):
            # Ensure each patient is assigned exactly one facility for treatment
            model.addCons(quicksum(alloc_resource[p, f] for f in range(n_facilities)) == 1, name=f"patient_alloc_resource_{p}")
            model.addCons(quicksum(alloc_staff[p, f] for f in range(n_facilities)) == 1, name=f"patient_alloc_staff_{p}")
            model.addCons(quicksum(travel[p, f] for f in range(n_facilities)) == 1, name=f"patient_travel_{p}")

            for f in range(n_facilities):
                # Only allow travel if resources and staff are allocated
                model.addCons(travel[p, f] <= alloc_resource[p, f], name=f"travel_resource_cons_{p}_{f}")
                model.addCons(travel[p, f] <= alloc_staff[p, f], name=f"travel_staff_cons_{p}_{f}")

        for f in range(n_facilities):
            for r in range(self.n_resources):
                # Do not exceed facility resource capacities
                model.addCons(quicksum(patients['resource_demand'][p, f] * alloc_resource[p, f] for p in range(n_patients)) <= facilities['capacity_resources'][f, r], name=f"facility_resource_capacity_{f}_{r}")
            for s in range(self.n_staff):
                # Do not exceed facility staff capacities
                model.addCons(quicksum(patients['staff_demand'][p, f] * alloc_staff[p, f] for p in range(n_patients)) <= facilities['capacity_staff'][f, s], name=f"facility_staff_capacity_{f}_{s}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facilities': 15,
        'max_facilities': 30,
        'min_patients': 30,
        'max_patients': 120,
        'n_resources': 9,
        'n_staff': 12,
    }

    resource_alloc = MedicalResourceAllocation(parameters, seed=seed)
    instance = resource_alloc.generate_instances()
    solve_status, solve_time = resource_alloc.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")