import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class HealthcareResourceManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def uniform(self, size, interval):
        return np.random.uniform(interval[0], interval[1], size)

    def generate_instance(self):
        hospital_capacities = self.randint(self.n_hospitals, self.capacity_interval)
        deployment_costs = self.uniform((self.n_neighborhoods, self.n_hospitals), self.cost_interval)
        neighborhood_health_needs = self.uniform(self.n_neighborhoods, self.need_interval)
        clinician_hiring_costs = self.randint(self.n_hospitals, self.hiring_cost_interval)

        res = {
            'hospital_capacities': hospital_capacities,
            'deployment_costs': deployment_costs,
            'neighborhood_health_needs': neighborhood_health_needs,
            'clinician_hiring_costs': clinician_hiring_costs
        }

        ### given instance data code ends
        ### new instance data code ends
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        hospital_capacities = instance['hospital_capacities']
        deployment_costs = instance['deployment_costs']
        neighborhood_health_needs = instance['neighborhood_health_needs']
        clinician_hiring_costs = instance['clinician_hiring_costs']

        n_neighborhoods = len(neighborhood_health_needs)
        n_hospitals = len(hospital_capacities)

        model = Model("HealthcareResourceManagement")

        # Decision variables
        new_hospital_unit = {
            (i, j): model.addVar(vtype="B", name=f"NewHospitalUnit_{i}_{j}")
            for i in range(n_neighborhoods) for j in range(n_hospitals)
        }
        hire_clinician = {j: model.addVar(vtype="B", name=f"HireClinician_{j}") for j in range(n_hospitals)}

        # Objective: minimize the total deployment and hiring cost
        objective_expr = (
            quicksum(deployment_costs[i, j] * new_hospital_unit[i, j] for i in range(n_neighborhoods) for j in range(n_hospitals)) +
            quicksum(clinician_hiring_costs[j] * hire_clinician[j] for j in range(n_hospitals)) +
            quicksum(neighborhood_health_needs[i] * quicksum(new_hospital_unit[i, j] for j in range(n_hospitals)) for i in range(n_neighborhoods))
        )

        # Constraints: Ensuring that each neighborhood is covered
        for i in range(n_neighborhoods):
            model.addCons(quicksum(new_hospital_unit[i, j] for j in range(n_hospitals)) >= 1, f"NeighborhoodCovered_{i}")

        # Constraints: Capacity limits for each hospital
        for j in range(n_hospitals):
            model.addCons(
                quicksum(new_hospital_unit[i, j] for i in range(n_neighborhoods)) <= hospital_capacities[j] * hire_clinician[j],
                f"Capacity_{j}"
            )

        # Constraints: Each hospital can only be used if it is hiring clinicians
        for i in range(n_neighborhoods):
            for j in range(n_hospitals):
                model.addCons(new_hospital_unit[i, j] <= hire_clinician[j], f"HospitalDeployed_{i}_{j}")

        ### Additional constraints for cost management ###
        # Constraint: limiting the maximum allowable deployment and hiring cost
        max_cost_budget = self.max_cost_budget
        model.addCons(
            quicksum(deployment_costs[i, j] * new_hospital_unit[i, j] for i in range(n_neighborhoods) for j in range(n_hospitals)) +
            quicksum(clinician_hiring_costs[j] * hire_clinician[j] for j in range(n_hospitals)) <= max_cost_budget,
            "MaxCostBudget"
        )

        ### given constraints and variables and objective code ends
        ### new constraints and variables and objective code ends
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 12345
    parameters = {
        'n_neighborhoods': 200,
        'n_hospitals': 37,
        'capacity_interval': (90, 900),
        'cost_interval': (150.0, 1500.0),
        'need_interval': (0.52, 0.52),
        'hiring_cost_interval': (50, 5000),
        'max_cost_budget': 800000.0,
    }
    ### given parameter code ends
    ### new parameter code ends

    deployment = HealthcareResourceManagement(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")