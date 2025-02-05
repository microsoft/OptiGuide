import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocationProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_factories > 0 and self.n_customers > 0
        assert self.min_setup_cost >= 0 and self.max_setup_cost >= self.min_setup_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost

        setup_costs = np.random.randint(self.min_setup_cost, self.max_setup_cost + 1, self.n_factories)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_factories, self.n_customers))

        demands = np.random.poisson(lam=self.mean_demand, size=self.n_customers)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_factories)

        return {
            "setup_costs": setup_costs,
            "transport_costs": transport_costs,
            "demands": demands,
            "capacities": capacities
        }

    # MILP modeling
    def solve(self, instance):
        setup_costs = instance["setup_costs"]
        transport_costs = instance["transport_costs"]
        demands = instance["demands"]
        capacities = instance["capacities"]

        model = Model("FacilityLocationProblem")
        n_factories = len(capacities)
        n_customers = len(demands)

        MaxLimit = self.max_limit

        # Decision variables
        open_factory = {f: model.addVar(vtype="B", name=f"Factory_{f}") for f in range(n_factories)}
        assign_customer = {(f, c): model.addVar(vtype="B", name=f"Factory_{f}_Customer_{c}") for f in range(n_factories) for c in range(n_customers)}

        # Objective function
        model.setObjective(
            quicksum(setup_costs[f] * open_factory[f] for f in range(n_factories)) +
            quicksum(transport_costs[f, c] * assign_customer[f, c] for f in range(n_factories) for c in range(n_customers)),
            "minimize"
        )

        # Constraints: Each customer must be fully served
        for c in range(n_customers):
            model.addCons(quicksum(assign_customer[f, c] for f in range(n_factories)) >= 1, f"Customer_{c}_Demand")

        # Constraints: Each factory cannot exceed its capacity
        for f in range(n_factories):
            model.addCons(quicksum(demands[c] * assign_customer[f, c] for c in range(n_customers)) <= capacities[f], f"Factory_{f}_Capacity")

        # Constraints: Only open factories can serve customers
        for f in range(n_factories):
            for c in range(n_customers):
                model.addCons(assign_customer[f, c] <= open_factory[f], f"Factory_{f}_Serve_{c}")

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
        'n_factories': 210,
        'n_customers': 50,
        'min_setup_cost': 2000,
        'max_setup_cost': 5000,
        'min_transport_cost': 10,
        'max_transport_cost': 500,
        'mean_demand': 25,
        'min_capacity': 500,
        'max_capacity': 5000,
        'max_limit': 5000,
    }

    factory_optimizer = FacilityLocationProblem(parameters, seed)
    instance = factory_optimizer.generate_instance()
    solve_status, solve_time, objective_value = factory_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")