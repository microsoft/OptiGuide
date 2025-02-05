import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimplifiedMicrochipManufacturing:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data generation #################
    def generate_instance(self):
        assert self.min_cost >= 0 and self.max_cost >= self.min_cost

        costs = self.min_cost + (self.max_cost - self.min_cost) * np.random.rand(self.n_components)
        compatibilities = np.triu(np.random.rand(self.n_components, self.n_components), k=1)
        compatibilities += compatibilities.transpose()
        compatibilities /= compatibilities.sum(1)

        orders = []

        for _ in range(self.n_orders):
            interests = np.random.rand(self.n_components)
            cost = costs.sum() / self.n_components  # simplified cost calculation

            config_mask = np.zeros(self.n_components)
            component = np.random.choice(self.n_components, p=interests / interests.sum())
            config_mask[component] = 1

            while np.random.rand() < self.add_component_prob:
                if config_mask.sum() == self.n_components:
                    break
                prob = (1 - config_mask) * compatibilities[config_mask.astype(bool), :].mean(axis=0)
                prob /= prob.sum()
                component = np.random.choice(self.n_components, p=prob)
                config_mask[component] = 1

            config = np.nonzero(config_mask)[0]
            orders.append((list(config), cost))

        orders_per_component = [[] for _ in range(self.n_components)]
        for i, order in enumerate(orders):
            config, cost = order
            for component in config:
                orders_per_component[component].append(i)

        dependencies = np.random.binomial(1, self.dependency_prob, size=(self.n_orders, self.n_orders))

        return {
            "orders": orders,
            "orders_per_component": orders_per_component,
            "dependencies": dependencies
        }
    
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        orders = instance['orders']
        orders_per_component = instance['orders_per_component']
        dependencies = instance['dependencies']
        
        model = Model("SimplifiedMicrochipManufacturing")
        
        # Decision variables
        order_vars = {i: model.addVar(vtype="B", name=f"Order_{i}") for i in range(len(orders))}
        penalty_vars = {i: model.addVar(vtype="B", name=f"Penalty_{i}") for i in range(len(orders))}
        
        # Objective: maximize the total profit and minimize penalties for unmet dependencies
        objective_expr = quicksum(order_vars[i] * cost - penalty_vars[i] * self.penalty_cost for i, (config, cost) in enumerate(orders))
        
        # Constraints: Each component can be in at most one configuration
        for component, order_indices in enumerate(orders_per_component):
            model.addCons(quicksum(order_vars[order_idx] for order_idx in order_indices) <= 1, f"Component_{component}")
        
        # Constraints: Dependencies between orders
        for i in range(len(orders)):
            for j in range(len(orders)):
                if dependencies[i, j] == 1:
                    model.addCons(order_vars[i] <= order_vars[j] + penalty_vars[i], f"Dependency_{i}_{j}")

        model.setObjective(objective_expr, "maximize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_components': 2500,
        'n_orders': 240,
        'min_cost': 480,
        'max_cost': 3000,
        'add_component_prob': 0.56,
        'dependency_prob': 0.52,
        'penalty_cost': 100,
    }

    manufacturing = SimplifiedMicrochipManufacturing(parameters, seed=seed)
    instance = manufacturing.generate_instance()
    solve_status, solve_time = manufacturing.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")