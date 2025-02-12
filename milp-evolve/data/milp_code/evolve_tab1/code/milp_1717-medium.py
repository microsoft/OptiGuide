import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        zones = range(self.number_of_zones)
        warehouses = range(self.potential_warehouses)

        # Generating zone parcel demand
        parcel_demand = np.random.gamma(shape=self.demand_shape, scale=self.demand_scale, size=self.number_of_zones).astype(int)
        parcel_demand = np.clip(parcel_demand, self.min_demand, self.max_demand)

        # Generating the delivery times
        delivery_times = np.random.beta(self.alpha_delivery_time, self.beta_delivery_time, size=self.number_of_zones)

        # Distance matrix between zones and potential warehouse locations
        G = nx.barabasi_albert_graph(self.number_of_zones, self.number_of_graph_edges)
        distances = nx.floyd_warshall_numpy(G) * np.random.uniform(self.min_distance_multiplier, self.max_distance_multiplier)

        # Generating warehouse costs and capacities
        warehouse_costs = np.random.gamma(self.warehouse_cost_shape, self.warehouse_cost_scale, size=self.potential_warehouses)
        warehouse_capacities = np.random.gamma(self.warehouse_capacity_shape, self.warehouse_capacity_scale, size=self.potential_warehouses).astype(int)

        res = {
            'parcel_demand': parcel_demand,
            'delivery_times': delivery_times,
            'distances': distances,
            'warehouse_costs': warehouse_costs,
            'warehouse_capacities': warehouse_capacities,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        parcel_demand = instance['parcel_demand']
        delivery_times = instance['delivery_times']
        distances = instance['distances']
        warehouse_costs = instance['warehouse_costs']
        warehouse_capacities = instance['warehouse_capacities']

        zones = range(len(parcel_demand))
        warehouses = range(len(warehouse_costs))

        model = Model("DeliveryOptimization")
        x = {}  # Binary variable: 1 if warehouse is placed at location l
        y = {}  # Binary variable: 1 if a delivery route from z to l exists
        f = {}  # Integer variable: Delivery frequency
        d = {}  # Continuous variable: Fraction of demand satisfied

        # Decision variables
        for l in warehouses:
            x[l] = model.addVar(vtype="B", name=f"x_{l}")
            for z in zones:
                y[z, l] = model.addVar(vtype="B", name=f"y_{z}_{l}")
                f[z, l] = model.addVar(vtype="I", name=f"f_{z}_{l}", lb=0, ub=self.max_delivery_frequency)
                d[z, l] = model.addVar(vtype="C", name=f"d_{z}_{l}", lb=0, ub=1)

        # Objective: Minimize delivery cost + warehouse costs + delivery impact + maximize delivery satisfaction
        obj_expr = quicksum(distances[z][l] * y[z, l] * parcel_demand[z] * delivery_times[z] for z in zones for l in warehouses) + \
                   quicksum(warehouse_costs[l] * x[l] for l in warehouses) + \
                   quicksum((1 - d[z, l]) * parcel_demand[z] for z in zones for l in warehouses) - \
                   quicksum(d[z, l] for z in zones for l in warehouses)

        model.setObjective(obj_expr, "minimize")

        # Constraints: Each zone must have access to one delivery route
        for z in zones:
            model.addCons(
                quicksum(y[z, l] for l in warehouses) == 1,
                f"Access_{z}"
            )

        # Constraints: Warehouse capacity must not be exceeded
        for l in warehouses:
            model.addCons(
                quicksum(y[z, l] * parcel_demand[z] * delivery_times[z] * d[z, l] for z in zones) <= warehouse_capacities[l] * x[l],
                f"Capacity_{l}"
            )

        # Constraints: Delivery frequency related to demand satisfaction
        for z in zones:
            for l in warehouses:
                model.addCons(
                    f[z, l] == parcel_demand[z] * d[z, l],
                    f"Frequency_Demand_{z}_{l}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_zones': 120,
        'potential_warehouses': 60,
        'demand_shape': 96,
        'demand_scale': 2400,
        'min_demand': 2500,
        'max_demand': 3000,
        'alpha_delivery_time': 378.0,
        'beta_delivery_time': 56.25,
        'number_of_graph_edges': 30,
        'min_distance_multiplier': 14,
        'max_distance_multiplier': 360,
        'warehouse_cost_shape': 50.0,
        'warehouse_cost_scale': 2000.0,
        'warehouse_capacity_shape': 300.0,
        'warehouse_capacity_scale': 2000.0,
        'max_delivery_frequency': 50,
    }

    delivery_optimization = DeliveryOptimization(parameters, seed=seed)
    instance = delivery_optimization.generate_instance()
    solve_status, solve_time = delivery_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")