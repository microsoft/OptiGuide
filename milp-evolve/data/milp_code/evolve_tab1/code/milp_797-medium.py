import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MicrochipManufacturing:
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

        def choose_next_component(config_mask, compatibilities):
            prob = (1 - config_mask) * compatibilities[config_mask, :].mean(axis=0)
            prob /= prob.sum()
            return np.random.choice(self.n_components, p=prob)

        costs = self.min_cost + (self.max_cost - self.min_cost) * np.random.rand(self.n_components)
        compatibilities = np.triu(np.random.rand(self.n_components, self.n_components), k=1)
        compatibilities += compatibilities.transpose()
        compatibilities /= compatibilities.sum(1)

        orders = []
        while len(orders) < self.n_orders:
            interests = np.random.rand(self.n_components)
            private_costs = costs + self.max_cost * self.cost_deviation * (2 * interests - 1)

            order_configs = {}
            prob = interests / interests.sum()
            component = np.random.choice(self.n_components, p=prob)
            config_mask = np.full(self.n_components, 0)
            config_mask[component] = 1

            while np.random.rand() < self.add_component_prob:
                if config_mask.sum() == self.n_components:
                    break
                component = choose_next_component(config_mask, compatibilities)
                config_mask[component] = 1

            config = np.nonzero(config_mask)[0]
            cost = private_costs[config].sum() + np.power(len(config), 1 + self.additivity)

            if cost < 0:
                continue

            order_configs[frozenset(config)] = cost

            sub_candidates = []
            for component in config:
                config_mask = np.full(self.n_components, 0)
                config_mask[component] = 1

                while config_mask.sum() < len(config):
                    component = choose_next_component(config_mask, compatibilities)
                    config_mask[component] = 1

                sub_config = np.nonzero(config_mask)[0]
                sub_cost = private_costs[sub_config].sum() + np.power(len(sub_config), 1 + self.additivity)
                sub_candidates.append((sub_config, sub_cost))

            budget = self.budget_factor * cost
            min_resale_value = self.resale_factor * costs[config].sum()
            for config, cost in [
                sub_candidates[i] for i in np.argsort([-cost for config, cost in sub_candidates])]:

                if len(order_configs) >= self.max_n_sub_orders + 1 or len(orders) + len(order_configs) >= self.n_orders:
                    break

                if cost < 0 or cost > budget:
                    continue

                if costs[config].sum() < min_resale_value:
                    continue

                if frozenset(config) in order_configs:
                    continue

                order_configs[frozenset(config)] = cost

            for config, cost in order_configs.items():
                orders.append((list(config), cost))

        orders_per_component = [[] for component in range(self.n_components)]
        for i, order in enumerate(orders):
            config, cost = order
            for component in config:
                orders_per_component[component].append(i)

        return {
            "orders": orders,
            "orders_per_component": orders_per_component
        }
    
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        orders = instance['orders']
        orders_per_component = instance['orders_per_component']
        
        model = Model("MicrochipManufacturing")
        
        # Decision variables
        order_vars = {i: model.addVar(vtype="B", name=f"Order_{i}") for i in range(len(orders))}
        
        # Objective: maximize the total profit
        objective_expr = quicksum(cost * order_vars[i] for i, (config, cost) in enumerate(orders))
        
        # Constraints: Each component can be in at most one configuration
        for component, order_indices in enumerate(orders_per_component):
            model.addCons(quicksum(order_vars[order_idx] for order_idx in order_indices) <= 1, f"Component_{component}")
        
        # Manufacturing capacity constraints
        max_capacity = self.max_capacity
        capacity_vars = [model.addVar(vtype="C", name=f"Capacity_{i}") for i in range(len(orders))]
        model.addCons(quicksum(capacity_vars) <= max_capacity, "Max_Capacity")

        model.setObjective(objective_expr, "maximize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_components': 225,
        'n_orders': 900,
        'min_cost': 960,
        'max_cost': 3000,
        'cost_deviation': 0.8,
        'additivity': 0.1,
        'add_component_prob': 0.8,
        'budget_factor': 459.38,
        'resale_factor': 0.38,
        'max_n_sub_orders': 600,
        'max_capacity': 5000,
    }

    manufacturing = MicrochipManufacturing(parameters, seed=42)
    instance = manufacturing.generate_instance()
    solve_status, solve_time = manufacturing.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")