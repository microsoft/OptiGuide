import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DiverseHealthCareNetworkOptimization:
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

        # Generate a random graph to model regional connections
        graph = nx.barabasi_albert_graph(self.nRegions, 2)
        transportCosts = nx.to_numpy_array(graph)
        transportCosts *= np.random.randint(20, 100)  # Scale transport costs

        return {
            "healthcareOperatingCosts": healthcareOperatingCosts,
            "regionDemands": regionDemands,
            "serviceTimes": serviceTimes,
            "qualityPenalties": qualityPenalties,
            "staff_requirements": staff_requirements,
            "transportCosts": transportCosts
        }

    def solve(self, instance):
        healthcareOperatingCosts = instance['healthcareOperatingCosts']
        regionDemands = instance['regionDemands']
        serviceTimes = instance['serviceTimes']
        qualityPenalties = instance['qualityPenalties']
        staff_requirements = instance['staff_requirements']
        transportCosts = instance['transportCosts']

        model = Model("DiverseHealthCareNetworkOptimization")

        nHealthcareCenters = len(healthcareOperatingCosts)
        nRegions = len(regionDemands)

        healthcare_center_vars = {h: model.addVar(vtype="B", name=f"HealthcareCenter_{h}") for h in range(nHealthcareCenters)}
        region_assignments = {(r, h): model.addVar(vtype="B", name=f"Assignment_{r}_{h}") for r in range(nRegions) for h in range(nHealthcareCenters)}
        transportation_vars = {(r1, r2): model.addVar(vtype="C", name=f"Transport_{r1}_{r2}") for r1 in range(nRegions) for r2 in range(nRegions)}

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

        for r1 in range(nRegions):
            for r2 in range(nRegions):
                if r1 != r2:
                    model.addCons(
                        transportation_vars[r1, r2] >= regionDemands[r1] * transportCosts[r1, r2],
                        f"TransportDemand_{r1}_{r2}"
                    )

        # Objective: Minimize cost and service time while considering service quality and transportation costs
        model.setObjective(
            quicksum(healthcareOperatingCosts[h] * healthcare_center_vars[h] for h in range(nHealthcareCenters)) +
            quicksum(serviceTimes[h, r] * region_assignments[r, h] for r in range(nRegions) for h in range(nHealthcareCenters)) +
            quicksum(transportCosts[r1, r2] * transportation_vars[r1, r2] for r1 in range(nRegions) for r2 in range(nRegions)),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    parameters = {
        'nHealthcareCenters': 50,
        'nRegions': 140,
        'maximumStaff': 3000,
        'qualityCostFactor': 250,
        'serviceCompliance': 0.85,
        'transportCostLimit': 10000,
    }
    seed = 42

    hc_solver = DiverseHealthCareNetworkOptimization(parameters, seed=seed)
    instance = hc_solver.get_instance()
    solve_status, solve_time, objective_value = hc_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")