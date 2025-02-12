import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HospitalResourceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_resources > 0 and self.n_hospitals > 0
        assert self.min_procure_cost >= 0 and self.max_procure_cost >= self.min_procure_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost

        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, self.n_resources)
        procure_costs = np.random.randint(self.min_procure_cost, self.max_procure_cost + 1, self.n_resources)
        needs = np.random.normal(self.mean_needs, self.std_dev_needs, self.n_hospitals).astype(int)
        capacities = np.random.randint(self.min_resource_capacity, self.max_resource_capacity + 1, self.n_resources)

        return {
            "transport_costs": transport_costs,
            "procure_costs": procure_costs,
            "needs": needs,
            "capacities": capacities
        }

    # MILP modeling
    def solve(self, instance):
        transport_costs = instance["transport_costs"]
        procure_costs = instance["procure_costs"]
        needs = instance["needs"]
        capacities = instance["capacities"]

        model = Model("HospitalResourceOptimization")
        n_resources = len(capacities)
        n_hospitals = len(needs)

        # Decision variables
        resource_allocated = {r: model.addVar(vtype="B", name=f"Resource_{r}") for r in range(n_resources)}
        hospital_supplies = {(r, h): model.addVar(vtype="C", name=f"Resource_{r}_Hospital_{h}") for r in range(n_resources) for h in range(n_hospitals)}
        unmet_needs = {h: model.addVar(vtype="C", name=f"Unmet_Needs_Hospital_{h}") for h in range(n_hospitals)}

        # Objective function: Minimize total cost
        model.setObjective(
            quicksum(transport_costs[r] * resource_allocated[r] for r in range(n_resources)) +
            quicksum(procure_costs[r] * hospital_supplies[r, h] for r in range(n_resources) for h in range(n_hospitals)) +
            self.penalty_unmet_needs * quicksum(unmet_needs[h] for h in range(n_hospitals)),
            "minimize"
        )

        # Constraints: Each hospital must meet its needs
        for h in range(n_hospitals):
            model.addCons(
                quicksum(hospital_supplies[r, h] for r in range(n_resources)) + unmet_needs[h] >= needs[h],
                f"Hospital_{h}_Needs"
            )

        # Constraints: Each resource cannot exceed its capacity
        for r in range(n_resources):
            model.addCons(
                quicksum(hospital_supplies[r, h] for h in range(n_hospitals)) <= capacities[r], 
                f"Resource_{r}_Capacity"
            )

        # Constraints: Only allocated resources can be used for supplies
        for r in range(n_resources):
            for h in range(n_hospitals):
                model.addCons(
                    hospital_supplies[r, h] <= capacities[r] * resource_allocated[r],
                    f"Resource_{r}_Supply_{h}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == "optimal":
            objective_value = model.getObjVal()
        else:
            objective_value = None

        return model.getStatus(), end_time - start_time, objective_value

if __name__ == "__main__":
    seed = 42
    parameters = {
        'n_resources': 525,
        'n_hospitals': 112,
        'min_transport_cost': 30,
        'max_transport_cost': 1500,
        'mean_needs': 2000,
        'std_dev_needs': 1500,
        'min_resource_capacity': 500,
        'max_resource_capacity': 3500,
        'min_procure_cost': 200,
        'max_procure_cost': 1250,
        'penalty_unmet_needs': 500,
    }

    resource_optimizer = HospitalResourceOptimization(parameters, seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")