import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DataCenterOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def get_instance(self):
        assert self.nHubs > 0 and self.nZones > 0
        assert self.maximumBandwidth > 0 and self.networkReliabilityCost >= 0

        hubOperatingCosts = np.random.randint(20000, 70000, self.nHubs)
        zoneDemands = np.random.randint(20, 200, self.nZones)
        
        communicationCosts = np.abs(np.random.normal(loc=50, scale=10, size=(self.nHubs, self.nZones)))
        communicationCosts = np.where(communicationCosts < 1, 1, communicationCosts)

        reliabilityPenalty = np.random.randint(50, 500, self.nZones)

        return {
            "hubOperatingCosts": hubOperatingCosts,
            "zoneDemands": zoneDemands,
            "communicationCosts": communicationCosts,
            "reliabilityPenalty": reliabilityPenalty
        }

    def solve(self, instance):
        hubOperatingCosts = instance['hubOperatingCosts']
        zoneDemands = instance['zoneDemands']
        communicationCosts = instance['communicationCosts']
        reliabilityPenalty = instance['reliabilityPenalty']

        model = Model("DataCenterOptimization")

        nHubs = len(hubOperatingCosts)
        nZones = len(zoneDemands)

        hub_vars = {h: model.addVar(vtype="B", name=f"Hub_{h}") for h in range(nHubs)}
        zone_assignments = {(z, h): model.addVar(vtype="B", name=f"Assignment_{z}_{h}") for z in range(nZones) for h in range(nHubs)}

        for z in range(nZones):
            model.addCons(
                quicksum(zone_assignments[z, h] for h in range(nHubs)) == 1,
                f"ZoneAssignment_{z}"
            )

        for h in range(nHubs):
            model.addCons(
                quicksum(zoneDemands[z] * zone_assignments[z, h] for z in range(nZones)) <= self.maximumBandwidth * hub_vars[h],
                f"HubCapacity_{h}"
            )

        for z in range(nZones):
            for h in range(nHubs):
                model.addCons(
                    zone_assignments[z, h] <= hub_vars[h],
                    f"AssignmentLimit_{z}_{h}"
                )

        for h in range(nHubs - 1):
            model.addCons(
                hub_vars[h] >= hub_vars[h + 1],
                f"Symmetry_{h}"
            )

        for z in range(nZones):
            model.addCons(
                quicksum(communicationCosts[h, z] * zone_assignments[z, h] for h in range(nHubs)) <= self.networkReliabilityCost + reliabilityPenalty[z],
                f"NetworkReliability_{z}"
            )

        model.setObjective(
            quicksum(hubOperatingCosts[h] * hub_vars[h] for h in range(nHubs)) +
            quicksum(communicationCosts[h, z] * zone_assignments[z, h] for z in range(nZones) for h in range(nHubs)),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    parameters = {
        'nHubs': 120,
        'nZones': 50,
        'maximumBandwidth': 2000,
        'networkReliabilityCost': 360,
    }

    seed = 42

    data_center_solver = DataCenterOptimization(parameters, seed=seed)
    instance = data_center_solver.get_instance()
    solve_status, solve_time, objective_value = data_center_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")