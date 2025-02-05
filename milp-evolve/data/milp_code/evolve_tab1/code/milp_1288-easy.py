import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FleetManagementOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def get_instance(self):
        assert self.NumberOfTrucks > 0 and self.RoutesPerNetwork > 0
        assert self.TransportCostRange[0] >= 0 and self.TransportCostRange[1] >= self.TransportCostRange[0]
        assert self.TruckCapacityRange[0] > 0 and self.TruckCapacityRange[1] >= self.TruckCapacityRange[0]

        operational_costs = np.random.randint(self.TransportCostRange[0], self.TransportCostRange[1] + 1, self.NumberOfTrucks)
        transport_costs = np.random.normal(loc=100, scale=15, size=(self.NumberOfTrucks, self.RoutesPerNetwork))
        transport_times = np.random.uniform(1, 5, size=(self.NumberOfTrucks, self.RoutesPerNetwork))
        capacities = np.random.randint(self.TruckCapacityRange[0], self.TruckCapacityRange[1] + 1, self.NumberOfTrucks)
        route_demands = np.random.randint(self.RouteDemandRange[0], self.RouteDemandRange[1] + 1, self.RoutesPerNetwork)
        time_windows = np.random.randint(50, 200, size=self.RoutesPerNetwork)
        speed_limits = np.random.uniform(40, 80, self.NumberOfTrucks)
        fuel_usage_rates = np.random.uniform(2, 8, self.NumberOfTrucks)

        return {
            "operational_costs": operational_costs,
            "transport_costs": transport_costs,
            "transport_times": transport_times,
            "capacities": capacities,
            "route_demands": route_demands,
            "time_windows": time_windows,
            "speed_limits": speed_limits,
            "fuel_usage_rates": fuel_usage_rates,
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        transport_costs = instance['transport_costs']
        transport_times = instance['transport_times']
        capacities = instance['capacities']
        route_demands = instance['route_demands']
        time_windows = instance['time_windows']
        speed_limits = instance['speed_limits']
        fuel_usage_rates = instance['fuel_usage_rates']

        model = Model("FleetManagementOptimization")
        n_trucks = len(operational_costs)
        n_routes = len(route_demands)

        fleet_vars = {t: model.addVar(vtype="B", name=f"Fleet_{t}") for t in range(n_trucks)}
        network_assignment_vars = {(t, r): model.addVar(vtype="B", name=f"Truck_{t}_Route_{r}") for t in range(n_trucks) for r in range(n_routes)}
        max_transport_time = model.addVar(vtype="C", name="Max_Transport_Time")
        allocation_vars = {(t, r): model.addVar(vtype="C", name=f"Penalty_{t}_{r}") for t in range(n_trucks) for r in range(n_routes)}

        # Objective function
        model.setObjective(
            quicksum(operational_costs[t] * fleet_vars[t] for t in range(n_trucks)) +
            quicksum(transport_costs[t][r] * network_assignment_vars[t, r] for t in range(n_trucks) for r in range(n_routes)) +
            quicksum(allocation_vars[t, r] for t in range(n_trucks) for r in range(n_routes)) +
            max_transport_time * 10,
            "minimize"
        )

        # Constraints
        # Route demand satisfaction (total assignments must cover total demand)
        for r in range(n_routes):
            model.addCons(quicksum(network_assignment_vars[t, r] for t in range(n_trucks)) >= route_demands[r], f"Route_Demand_Satisfaction_{r}")

        # Capacity limits for each truck
        for t in range(n_trucks):
            model.addCons(quicksum(network_assignment_vars[t, r] for r in range(n_routes)) <= capacities[t] * fleet_vars[t], f"Fleet_Capacity_{t}")

        # Time window constraints with penalties for late deliveries
        for r in range(n_routes):
            for t in range(n_trucks):
                model.addCons(network_assignment_vars[t, r] * transport_times[t][r] <= time_windows[r] + allocation_vars[t, r], f"Time_Window_{t}_{r}")

        # Max transport time constraint
        for t in range(n_trucks):
            for r in range(n_routes):
                model.addCons(network_assignment_vars[t, r] * transport_times[t][r] <= max_transport_time, f"Max_Transport_Time_Constraint_{t}_{r}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'NumberOfTrucks': 100,
        'RoutesPerNetwork': 100,
        'TransportCostRange': (100, 300),
        'TruckCapacityRange': (450, 1800),
        'RouteDemandRange': (3, 15),
    }

    fleet_optimizer = FleetManagementOptimization(parameters, seed=seed)
    instance = fleet_optimizer.get_instance()
    solve_status, solve_time, objective_value = fleet_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")