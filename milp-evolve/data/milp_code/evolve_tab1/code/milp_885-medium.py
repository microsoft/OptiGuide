import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def delivery_times(self):
        base_delivery_time = 5.0  # base delivery time in units
        return base_delivery_time * np.random.rand(self.n_hubs, self.n_depots)

    def generate_instance(self):
        depot_demand = self.randint(self.n_depots, self.demand_interval)
        hub_supplies = self.randint(self.n_hubs, self.supply_interval)
        activation_costs = self.randint(self.n_hubs, self.activation_cost_interval)
        delivery_times = self.delivery_times()

        hub_supplies = hub_supplies * self.ratio * np.sum(depot_demand) / np.sum(hub_supplies)
        hub_supplies = np.round(hub_supplies)

        res = {
            'depot_demand': depot_demand,
            'hub_supplies': hub_supplies,
            'activation_costs': activation_costs,
            'delivery_times': delivery_times,
        }
        return res

    def solve(self, instance):
        depot_demand = instance['depot_demand']
        hub_supplies = instance['hub_supplies']
        activation_costs = instance['activation_costs']
        delivery_times = instance['delivery_times']

        n_depots = len(depot_demand)
        n_hubs = len(hub_supplies)

        model = Model("SupplyChainNetwork")

        # Decision variables
        activate_hub = {j: model.addVar(vtype="B", name=f"Activate_{j}") for j in range(n_hubs)}
        assign_delivery = {(j, k): model.addVar(vtype="B", name=f"Assign_{j}_{k}") for k in range(n_depots) for j in range(n_hubs)}

        # Objective: Minimize the total cost including activation and transportation
        transportation_cost_per_time = 20
        objective_expr = quicksum(activation_costs[j] * activate_hub[j] for j in range(n_hubs)) + \
                         transportation_cost_per_time * quicksum(delivery_times[j, k] * assign_delivery[j, k] for j in range(n_hubs) for k in range(n_depots))

        # Constraints: each depot must be supplied by exactly one hub
        for k in range(n_depots):
            model.addCons(quicksum(assign_delivery[j, k] for j in range(n_hubs)) == 1, f"Depot_Supply_{k}")
        
        # Constraints: hub supply limits must be respected
        for j in range(n_hubs):
            model.addCons(quicksum(depot_demand[k] * assign_delivery[j, k] for k in range(n_depots)) <= hub_supplies[j] * activate_hub[j], f"Hub_Supply_{j}")

        # Constraint: Transportation times limited (deliveries within permissible limits)
        for j in range(n_hubs):
            for k in range(n_depots):
                model.addCons(delivery_times[j, k] * assign_delivery[j, k] <= activate_hub[j] * 50, f"Time_Limit_{j}_{k}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_depots': 50,
        'n_hubs': 120,
        'demand_interval': (30, 90),
        'supply_interval': (400, 1000),
        'activation_cost_interval': (1000, 3000),
        'ratio': 75.0,
    }

    supply_chain = SupplyChainNetwork(parameters, seed=seed)
    instance = supply_chain.generate_instance()
    solve_status, solve_time = supply_chain.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")