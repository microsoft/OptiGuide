import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EVDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_charging_stations > 0 and self.n_customers > 0
        assert self.min_energy_cost >= 0 and self.max_energy_cost >= self.min_energy_cost
        assert self.min_travel_cost >= 0 and self.max_travel_cost >= self.min_travel_cost
        assert self.min_battery_capacity > 0 and self.max_battery_capacity >= self.min_battery_capacity

        energy_costs = np.random.randint(self.min_energy_cost, self.max_energy_cost + 1, self.n_charging_stations)
        travel_costs = np.random.randint(self.min_travel_cost, self.max_travel_cost + 1, (self.n_charging_stations, self.n_customers))
        battery_capacities = np.random.randint(self.min_battery_capacity, self.max_battery_capacity + 1, self.n_charging_stations)
        customer_demands = np.random.randint(1, 10, self.n_customers)
        emission_limits = np.random.uniform(self.min_emission_limit, self.max_emission_limit, self.n_charging_stations)

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.n_charging_stations):
            for d in range(self.n_customers):
                G.add_edge(f"charging_{p}", f"customer_{d}")
                node_pairs.append((f"charging_{p}", f"customer_{d}"))

        return {
            "energy_costs": energy_costs,
            "travel_costs": travel_costs,
            "battery_capacities": battery_capacities,
            "customer_demands": customer_demands,
            "emission_limits": emission_limits,
            "graph": G,
            "node_pairs": node_pairs,
        }

    def solve(self, instance):
        energy_costs = instance['energy_costs']
        travel_costs = instance['travel_costs']
        battery_capacities = instance['battery_capacities']
        customer_demands = instance['customer_demands']
        emission_limits = instance['emission_limits']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("EVDeliveryOptimization")
        n_charging_stations = len(energy_costs)
        n_customers = len(travel_costs[0])

        # Decision variables
        ev_vars = {p: model.addVar(vtype="B", name=f"Dispatch_{p}") for p in range(n_charging_stations)}
        distance_vars = {(u, v): model.addVar(vtype="C", name=f"Distance_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total cost including energy costs and travel costs.
        model.setObjective(
            quicksum(energy_costs[p] * ev_vars[p] for p in range(n_charging_stations)) +
            quicksum(travel_costs[p, int(v.split('_')[1])] * distance_vars[(u, v)] for (u, v) in node_pairs for p in range(n_charging_stations) if u == f'charging_{p}'),
            "minimize"
        )

        # Distance constraint for each customer
        for d in range(n_customers):
            model.addCons(
                quicksum(distance_vars[(u, f"customer_{d}")] for u in G.predecessors(f"customer_{d}")) == customer_demands[d], 
                f"Customer_{d}_DistanceRequirements"
            )

        # Constraints: Customers only receive travel service if the stations are active and within battery limits
        for p in range(n_charging_stations):
            for d in range(n_customers):
                model.addCons(
                    distance_vars[(f"charging_{p}", f"customer_{d}")] <= emission_limits[p] * ev_vars[p], 
                    f"Charging_{p}_DistanceLimitByEmissions_{d}"
                )

        # Constraints: Charging stations cannot exceed their battery capacities
        for p in range(n_charging_stations):
            model.addCons(
                quicksum(distance_vars[(f"charging_{p}", f"customer_{d}")] for d in range(n_customers)) <= battery_capacities[p], 
                f"Charging_{p}_MaxBatteryCapacity"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_charging_stations': 240,
        'n_customers': 140,
        'min_travel_cost': 50,
        'max_travel_cost': 800,
        'min_energy_cost': 600,
        'max_energy_cost': 3000,
        'min_battery_capacity': 2000,
        'max_battery_capacity': 2400,
        'min_emission_limit': 200,
        'max_emission_limit': 2400,
    }

    optimizer = EVDeliveryOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")