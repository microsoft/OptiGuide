import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FactoryAllocationNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.min_demand >= 0 and self.max_demand >= self.min_demand
        assert self.min_capacity >= 0 and self.max_capacity >= self.min_capacity

        # Generate random factory capacities
        factory_capacities = self.min_capacity + (self.max_capacity - self.min_capacity) * np.random.rand(self.n_factories)
        # Generate random demands for each order
        order_demands = self.min_demand + (self.max_demand - self.min_demand) * np.random.rand(self.n_orders)
        # Generate random transportation costs
        transport_costs = np.random.rand(self.n_orders, self.n_factories)
        # Generate random factory operational costs
        factory_costs = np.random.exponential(50, size=self.n_factories).tolist()

        return {
            "factory_capacities": factory_capacities,
            "order_demands": order_demands,
            "transport_costs": transport_costs,
            "factory_costs": factory_costs
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        factory_capacities = instance['factory_capacities']
        order_demands = instance['order_demands']
        transport_costs = instance['transport_costs']
        factory_costs = instance['factory_costs']

        model = Model("FactoryAllocationNetworkOptimization")

        allocation_vars = {(i, j): model.addVar(vtype="C", name=f"alloc_{i}_{j}") for i in range(self.n_factories) for j in range(self.n_orders)}
        factory_usage_vars = {i: model.addVar(vtype="B", name=f"usage_{i}") for i in range(self.n_factories)}

        # Objective: minimize the total cost (manufacturing + transportation)
        objective_expr = (
            quicksum(factory_costs[i] * factory_usage_vars[i] for i in range(self.n_factories)) +
            quicksum(transport_costs[j][i] * allocation_vars[i, j] for i in range(self.n_factories) for j in range(self.n_orders))
        )
        
        # Add constraints
        # Constraint: All demands must be satisfied
        for j in range(self.n_orders):
            model.addCons(quicksum(allocation_vars[i, j] for i in range(self.n_factories)) == order_demands[j], f"Demand_{j}")

        # Constraint: Factory capacity must not be exceeded
        for i in range(self.n_factories):
            model.addCons(quicksum(allocation_vars[i, j] for j in range(self.n_orders)) <= factory_capacities[i] * factory_usage_vars[i], f"Capacity_{i}")

        # Constraint: Linking factory usage to allocation
        for i in range(self.n_factories):
            model.addCons(factory_usage_vars[i] <= 1, f"FactoryUsage_{i}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_factories': 200,
        'n_orders': 50,
        'min_demand': 20,
        'max_demand': 2000,
        'min_capacity': 900,
        'max_capacity': 3000,
    }

    factory_allocation_network = FactoryAllocationNetworkOptimization(parameters, seed=seed)
    instance = factory_allocation_network.generate_instance()
    solve_status, solve_time = factory_allocation_network.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")