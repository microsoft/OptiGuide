import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class ExtendedMicrochipManufacturing:
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

        resource_min_usage = np.random.uniform(self.resource_min_lower, self.resource_min_upper, size=self.n_orders)
        resource_max_usage = np.random.uniform(self.resource_max_lower, self.resource_max_upper, size=self.n_orders)
        
        G = self.generate_network_graph()
        commodity_types = ['A', 'B', 'C']
        transmission_costs = {(u, v, c): np.random.uniform(10, 50) for u, v in G.edges for c in commodity_types}
        flow_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}

        return {
            "orders": orders,
            "orders_per_component": orders_per_component,
            "dependencies": dependencies,
            "resource_min_usage": resource_min_usage,
            "resource_max_usage": resource_max_usage,
            "G": G,
            "flow_probabilities": flow_probabilities,
            "commodity_types": commodity_types,
            "transmission_costs": transmission_costs
        }

    def generate_network_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_orders, p=self.connection_prob, directed=True, seed=self.seed)
        return G

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        orders = instance['orders']
        orders_per_component = instance['orders_per_component']
        dependencies = instance['dependencies']
        resource_min_usage = instance['resource_min_usage']
        resource_max_usage = instance['resource_max_usage']
        G = instance['G']
        flow_probabilities = instance['flow_probabilities']
        commodity_types = instance['commodity_types']
        transmission_costs = instance['transmission_costs']

        model = Model("ExtendedMicrochipManufacturing")
        
        # Decision variables
        order_vars = {i: model.addVar(vtype="B", name=f"Order_{i}") for i in range(len(orders))}
        penalty_vars = {i: model.addVar(vtype="B", name=f"Penalty_{i}") for i in range(len(orders))}
        
        # New semi-continuous variables
        resource_vars = {i: model.addVar(lb=0, ub=resource_max_usage[i], name=f"Resource_{i}", vtype="C") for i in range(len(orders))}
        
        # New Variables for network flows and commodities
        network_flow_vars = {(u, v): model.addVar(vtype="B", name=f"NetworkFlow_{u}_{v}") for u, v in G.edges}
        commodity_flow_vars = {(u, v, c): model.addVar(vtype="C", name=f"CommodityFlow_{u}_{v}_{c}") for u, v in G.edges for c in commodity_types}
        
        # New logical activation variables using Big M
        activate_resource_vars = {i: model.addVar(vtype="B", name=f"ActivateResource_{i}") for i in range(len(orders))}
        dependency_vars = {(i, j): model.addVar(vtype="B", name=f"Dependency_{i}_{j}") for i in range(len(orders)) for j in range(len(orders)) if dependencies[i, j] == 1}

        M = 10**6  # Big M constant

        # Objective: maximize the total profit and minimize penalties for unmet dependencies
        objective_expr = quicksum(order_vars[i] * cost - penalty_vars[i] * self.penalty_cost for i, (config, cost) in enumerate(orders))

        # New Objective Component: Transmission Costs for Commodities
        objective_expr -= quicksum(transmission_costs[(u, v, c)] * commodity_flow_vars[(u, v, c)] for u, v in G.edges for c in commodity_types)
        
        for component, order_indices in enumerate(orders_per_component):
            model.addCons(quicksum(order_vars[order_idx] for order_idx in order_indices) <= 1, f"Component_{component}")
        
        for i in range(len(orders)):
            for j in range(len(orders)):
                if dependencies[i, j] == 1:
                    model.addCons(order_vars[i] <= order_vars[j] + penalty_vars[i], f"Dependency_{i}_{j}")

        for i in range(len(orders)):
            model.addCons(resource_vars[i] >= resource_min_usage[i] * order_vars[i], f"MinUsage_{i}")
            model.addCons(resource_vars[i] <= resource_max_usage[i] * order_vars[i], f"MaxUsage_{i}")

            # Big M constraints for resource activation
            model.addCons(resource_vars[i] <= M * activate_resource_vars[i], f"ActivateResource_{i}")
            model.addCons(order_vars[i] <= activate_resource_vars[i], name=f"OrderACT_{i}")

        for u, v in G.edges:
            model.addCons(network_flow_vars[(u, v)] <= flow_probabilities[(u, v)], name=f"FlowProbability_{u}_{v}")

        for c in commodity_types:
            for node in G.nodes:
                inflow = quicksum(commodity_flow_vars[(u, v, c)] for u, v in G.edges if v == node)
                outflow = quicksum(commodity_flow_vars[(u, v, c)] for u, v in G.edges if u == node)
                model.addCons(inflow - outflow == 0, name=f"FlowBalance_{node}_{c}")

        # Set objective
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_components': 1406,
        'n_orders': 240,
        'min_cost': 2400,
        'max_cost': 3000,
        'add_component_prob': 0.24,
        'dependency_prob': 0.8,
        'penalty_cost': 700,
        'resource_min_lower': 375,
        'resource_min_upper': 1050,
        'resource_max_lower': 1500,
        'resource_max_upper': 1875,
        'connection_prob': 0.31,
        'M': '10 ** 6',
    }

    manufacturing = ExtendedMicrochipManufacturing(parameters, seed=seed)
    instance = manufacturing.generate_instance()
    solve_status, solve_time = manufacturing.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")