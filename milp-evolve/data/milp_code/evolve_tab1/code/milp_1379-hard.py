import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedHealthCareNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def get_instance(self):
        assert self.nHealthcareCenters > 0 and self.nRegions > 0
        assert self.maximumStaff > 0 and self.qualityCostFactor >= 0

        healthcareOperatingCosts = np.random.gamma(shape=2.0, scale=20000, size=self.nHealthcareCenters)
        regionDemands = np.random.normal(loc=100, scale=20, size=self.nRegions).astype(int)
        regionDemands[regionDemands < 0] = 1  # Ensuring positive demand
        serviceTimes = np.random.normal(loc=80, scale=15, size=(self.nHealthcareCenters, self.nRegions))
        serviceTimes = np.where(serviceTimes < 1, 1, serviceTimes)
        qualityPenalties = np.random.randint(50, 300, self.nRegions)

        staff_requirements = np.random.randint(5, 15, (self.nHealthcareCenters, self.nRegions))

        return {
            "healthcareOperatingCosts": healthcareOperatingCosts,
            "regionDemands": regionDemands,
            "serviceTimes": serviceTimes,
            "qualityPenalties": qualityPenalties,
            "staff_requirements": staff_requirements
        }

    def solve(self, instance):
        healthcareOperatingCosts = instance['healthcareOperatingCosts']
        regionDemands = instance['regionDemands']
        serviceTimes = instance['serviceTimes']
        qualityPenalties = instance['qualityPenalties']
        staff_requirements = instance['staff_requirements']

        model = Model("SimplifiedHealthCareNetworkOptimization")

        nHealthcareCenters = len(healthcareOperatingCosts)
        nRegions = len(regionDemands)

        healthcare_center_vars = {h: model.addVar(vtype="B", name=f"HealthcareCenter_{h}") for h in range(nHealthcareCenters)}
        region_assignments = {(r, h): model.addVar(vtype="B", name=f"Assignment_{r}_{h}") for r in range(nRegions) for h in range(nHealthcareCenters)}

        for r in range(nRegions):
            model.addCons(
                quicksum(region_assignments[r, h] for h in range(nHealthcareCenters)) == 1,
                f"RegionCoverage_{r}"
            )

        for h in range(nHealthcareCenters):
            model.addCons(
                quicksum(regionDemands[r] * region_assignments[r, h] for r in range(nRegions)) <= self.maximumStaff * healthcare_center_vars[h],
                f"StaffCapacity_{h}"
            )

        for r in range(nRegions):
            for h in range(nHealthcareCenters):
                model.addCons(
                    region_assignments[r, h] <= healthcare_center_vars[h],
                    f"AssignmentLimit_{r}_{h}"
                )

        for r in range(nRegions):
            model.addCons(
                quicksum(serviceTimes[h, r] * region_assignments[r, h] for h in range(nHealthcareCenters)) <= self.qualityCostFactor + qualityPenalties[r],
                f"QualityService_{r}"
            )

        model.setObjective(
            quicksum(healthcareOperatingCosts[h] * healthcare_center_vars[h] for h in range(nHealthcareCenters)) +
            quicksum(serviceTimes[h, r] * region_assignments[r, h] for r in range(nRegions) for h in range(nHealthcareCenters)),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    parameters = {
        'nHealthcareCenters': 100,
        'nRegions': 140,
        'maximumStaff': 3000,
        'qualityCostFactor': 250,
    }
    seed = 42

    hc_solver = SimplifiedHealthCareNetworkOptimization(parameters, seed=seed)
    instance = hc_solver.get_instance()
    solve_status, solve_time, objective_value = hc_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")