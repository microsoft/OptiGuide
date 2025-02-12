import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class LogisticsNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def get_instance(self):
        assert self.nFacilities > 0 and self.nCustomers > 0
        assert self.facilityCapacityMax > 0 and self.assignmentTimePenalty >= 0

        facilityOperatingCosts = np.random.randint(10000, 50000, self.nFacilities)
        customerDemands = np.random.randint(5, 50, self.nCustomers)
        
        travelTimes = np.abs(np.random.normal(loc=120, scale=30, size=(self.nFacilities, self.nCustomers)))
        travelTimes = np.where(travelTimes < 1, 1, travelTimes)

        return {
            "facilityOperatingCosts": facilityOperatingCosts,
            "customerDemands": customerDemands,
            "travelTimes": travelTimes
        }

    def solve(self, instance):
        facilityOperatingCosts = instance['facilityOperatingCosts']
        customerDemands = instance['customerDemands']
        travelTimes = instance['travelTimes']

        model = Model("LogisticsNetworkOptimization")

        nFacilities = len(facilityOperatingCosts)
        nCustomers = len(customerDemands)

        facilities_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(nFacilities)}
        customer_assignments = {(c, f): model.addVar(vtype="B", name=f"Assignment_{c}_{f}") for c in range(nCustomers) for f in range(nFacilities)}

        for c in range(nCustomers):
            model.addCons(
                quicksum(customer_assignments[c, f] for f in range(nFacilities)) == 1,
                f"CustomerAssignment_{c}"
            )

        for f in range(nFacilities):
            model.addCons(
                quicksum(customerDemands[c] * customer_assignments[c, f] for c in range(nCustomers)) <= self.facilityCapacityMax * facilities_vars[f],
                f"FacilityCapacity_{f}"
            )

        for c in range(nCustomers):
            for f in range(nFacilities):
                model.addCons(
                    customer_assignments[c, f] <= facilities_vars[f],
                    f"AssignmentLimit_{c}_{f}"
                )

        for f in range(nFacilities - 1):
            model.addCons(
                facilities_vars[f] >= facilities_vars[f + 1],
                f"Symmetry_{f}"
            )

        for c in range(nCustomers):
            model.addCons(
                quicksum(travelTimes[f, c] * customer_assignments[c, f] for f in range(nFacilities)) <= self.complianceTolerance + (1 - self.complianceRate) * sum(travelTimes[:, c]),
                f"Compliance_{c}"
            )

        model.setObjective(
            quicksum(facilityOperatingCosts[f] * facilities_vars[f] for f in range(nFacilities)) +
            quicksum(travelTimes[f, c] * customer_assignments[c, f] for c in range(nCustomers) for f in range(nFacilities)),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'nFacilities': 80,
        'nCustomers': 120,
        'facilityCapacityMax': 1050,
        'assignmentTimePenalty': 900,
        'complianceRate': 0.86,
        'complianceTolerance': 2000,
    }

    logistics_solver = LogisticsNetworkOptimization(parameters, seed=42)
    instance = logistics_solver.get_instance()
    solve_status, solve_time, objective_value = logistics_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")