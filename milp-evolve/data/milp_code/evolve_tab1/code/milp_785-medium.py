import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper Function #############
class Healthcare:
    """
    Helper function: Container for healthcare context.
    """
    def __init__(self, number_of_doctors, number_of_patients, patient_doctor_pairs):
        self.number_of_doctors = number_of_doctors
        self.number_of_patients = number_of_patients
        self.patient_doctor_pairs = patient_doctor_pairs

    def generate_instance(self):
        pairs = set()
        capacities = np.random.randint(5, 20, size=self.number_of_doctors)

        for pair in self.patient_doctor_pairs:
            pairs.add(pair)
        
        res = {'number_of_doctors': self.number_of_doctors,
               'number_of_patients': self.number_of_patients,
               'patient_doctor_pairs': pairs,
               'capacities': capacities}

        return res

class PatientCoverage:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_patient_doctor_pairs(self):
        pairs = set()
        for patient in range(self.n_patients):
            num_pairs = np.random.randint(1, self.n_doctors // 2)
            doctors = np.random.choice(self.n_doctors, num_pairs, replace=False)
            for doctor in doctors:
                pairs.add((patient, doctor))
        return pairs

    def generate_instance(self):
        pairs = self.generate_patient_doctor_pairs()
        healthcare = Healthcare(self.n_doctors, self.n_patients, pairs)
        return healthcare.generate_instance()

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        number_of_doctors = instance['number_of_doctors']
        number_of_patients = instance['number_of_patients']
        patient_doctor_pairs = instance['patient_doctor_pairs']
        capacities = instance['capacities']
        
        model = Model("PatientCoverage")
        doctor_vars = {}
        patient_covered = {}

        for doctor in range(number_of_doctors):
            doctor_vars[doctor] = model.addVar(vtype="I", lb=0, ub=capacities[doctor], name=f"d_{doctor}")

        for patient in range(number_of_patients):
            patient_covered[patient] = model.addVar(vtype="B", name=f"p_{patient}")

        for patient, doctor in patient_doctor_pairs:
            model.addCons(patient_covered[patient] <= doctor_vars[doctor], name=f"coverage_{patient}_{doctor}")

        for doctor in range(number_of_doctors):
            model.addCons(quicksum(patient_covered[patient] for patient, doc in patient_doctor_pairs if doc == doctor) <= doctor_vars[doctor], name=f"capacity_{doctor}")

        objective_expr = quicksum(patient_covered[patient] for patient in range(number_of_patients))
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_doctors': 500,
        'n_patients': 1500,
    }

    patient_coverage_problem = PatientCoverage(parameters, seed=seed)
    instance = patient_coverage_problem.generate_instance()
    solve_status, solve_time = patient_coverage_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")