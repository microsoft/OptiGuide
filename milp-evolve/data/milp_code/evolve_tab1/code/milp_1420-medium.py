import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ManufacturingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def get_instance(self):
        assert self.nMachines > 0 and self.nProducts > 0
        assert self.maximumProductionCapacity > 0 and self.efficiencyFactor >= 0

        machineOperatingCosts = np.random.gamma(shape=2.0, scale=10000, size=self.nMachines)
        productDemands = np.random.normal(loc=150, scale=30, size=self.nProducts).astype(int)
        productDemands[productDemands < 0] = 1  # Ensure positive demand
        productionTimes = np.random.normal(loc=60, scale=10, size=(self.nMachines, self.nProducts))
        productionTimes = np.where(productionTimes < 1, 1, productionTimes)
        efficiencyPenalties = np.random.randint(20, 100, self.nProducts)
        machineEfficiencies = np.random.randint(70, 100, self.nMachines)

        return {
            "machineOperatingCosts": machineOperatingCosts,
            "productDemands": productDemands,
            "productionTimes": productionTimes,
            "efficiencyPenalties": efficiencyPenalties,
            "machineEfficiencies": machineEfficiencies
        }

    def solve(self, instance):
        machineOperatingCosts = instance['machineOperatingCosts']
        productDemands = instance['productDemands']
        productionTimes = instance['productionTimes']
        efficiencyPenalties = instance['efficiencyPenalties']
        machineEfficiencies = instance['machineEfficiencies']

        model = Model("ManufacturingOptimization")

        nMachines = len(machineOperatingCosts)
        nProducts = len(productDemands)

        machine_vars = {m: model.addVar(vtype="B", name=f"ManufacturingStatus_{m}") for m in range(nMachines)}
        product_assignments = {(p, m): model.addVar(vtype="B", name=f"Assignment_{p}_{m}") for p in range(nProducts) for m in range(nMachines)}

        for p in range(nProducts):
            model.addCons(
                quicksum(product_assignments[p, m] for m in range(nMachines)) == 1,
                f"ProductCoverage_{p}"
            )

        for m in range(nMachines):
            model.addCons(
                quicksum(productDemands[p] * product_assignments[p, m] for p in range(nProducts)) <= self.maximumProductionCapacity * machine_vars[m],
                f"CapacityConstraint_{m}"
            )

        for p in range(nProducts):
            for m in range(nMachines):
                model.addCons(
                    product_assignments[p, m] <= machine_vars[m],
                    f"AssignmentLimit_{p}_{m}"
                )

        for p in range(nProducts):
            model.addCons(
                quicksum(productionTimes[m, p] * product_assignments[p, m] for m in range(nMachines)) <= self.efficiencyFactor + efficiencyPenalties[p],
                f"EfficiencyConstraint_{p}"
            )

        model.setObjective(
            quicksum(productDemands[p] * machineEfficiencies[m] * product_assignments[p, m] for m in range(nMachines) for p in range(nProducts)),
            "maximize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    parameters = {
        'nMachines': 50,
        'nProducts': 52,
        'maximumProductionCapacity': 1500,
        'efficiencyFactor': 40,
    }
    seed = 42

    manufacturing_solver = ManufacturingOptimization(parameters, seed=seed)
    instance = manufacturing_solver.get_instance()
    solve_status, solve_time, objective_value = manufacturing_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")