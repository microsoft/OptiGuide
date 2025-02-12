import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class DroneDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_drones > 0 and self.n_regions > 0
        assert self.min_operational_cost >= 0 and self.max_operational_cost >= self.min_operational_cost
        assert self.min_delivery_cost >= 0 and self.max_delivery_cost >= self.min_delivery_cost
        assert self.min_battery_limit > 0 and self.max_battery_limit >= self.min_battery_limit

        operational_costs = np.random.randint(self.min_operational_cost, self.max_operational_cost + 1, self.n_drones)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_drones, self.n_regions))
        battery_limits = np.random.randint(self.min_battery_limit, self.max_battery_limit + 1, self.n_drones)
        region_urgencies = np.random.randint(1, 10, self.n_regions)
        demand_levels = np.random.uniform(self.min_demand_level, self.max_demand_level, self.n_regions)
        distances = np.random.uniform(0, self.max_delivery_distance, (self.n_drones, self.n_regions))
        
        region_positions = np.random.rand(self.n_regions, 2) * self.city_size
        drone_positions = np.random.rand(self.n_drones, 2) * self.city_size

        G = nx.DiGraph()
        node_pairs = []
        for d in range(self.n_drones):
            for r in range(self.n_regions):
                G.add_edge(f"drone_{d}", f"region_{r}")
                node_pairs.append((f"drone_{d}", f"region_{r}"))

        return {
            "operational_costs": operational_costs,
            "delivery_costs": delivery_costs,
            "battery_limits": battery_limits,
            "region_urgencies": region_urgencies,
            "demand_levels": demand_levels,
            "distances": distances,
            "region_positions": region_positions,
            "drone_positions": drone_positions,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        delivery_costs = instance['delivery_costs']
        battery_limits = instance['battery_limits']
        region_urgencies = instance['region_urgencies']
        demand_levels = instance['demand_levels']
        distances = instance['distances']
        region_positions = instance['region_positions']
        drone_positions = instance['drone_positions']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        
        model = Model("DroneDeliveryOptimization")
        n_drones = len(operational_costs)
        n_regions = len(delivery_costs[0])

        # Decision variables
        active_vars = {d: model.addVar(vtype="B", name=f"Drone_{d}") for d in range(n_drones)}
        route_vars = {(u, v): model.addVar(vtype="C", name=f"Route_{u}_{v}") for u, v in node_pairs}
        urgency_vars = {r: model.addVar(vtype="B", name=f"Urgency_Region_{r}") for r in range(n_regions)}

        # Objective: minimize the total cost including operational costs and delivery costs, and penalize late deliveries.
        model.setObjective(
            quicksum(operational_costs[d] * active_vars[d] for d in range(n_drones)) +
            quicksum(delivery_costs[d, int(v.split('_')[1])] * route_vars[(u, v)] for (u, v) in node_pairs for d in range(n_drones) if u == f'drone_{d}') +
            quicksum(region_urgencies[r] * urgency_vars[r] for r in range(n_regions)),
            "minimize"
        )

        # Battery limits for each drone
        for d in range(n_drones):
            model.addCons(
                quicksum(route_vars[(f"drone_{d}", f"region_{r}")] for r in range(n_regions)) <= battery_limits[d],
                f"DroneBatteryLimits_Drone_{d}"
            )

        # Minimum demand coverage for each region
        for r in range(n_regions):
            model.addCons(
                quicksum(route_vars[(u, f"region_{r}")] for u in G.predecessors(f"region_{r}")) >= demand_levels[r],
                f"MinimumDemandCoverage_Region_{r}"
            )

        # Efficient delivery for drones
        for d in range(n_drones):
            for r in range(n_regions):
                model.addCons(
                    route_vars[(f"drone_{d}", f"region_{r}")] <= self.efficiency_factor * active_vars[d],
                    f"EfficiencyDelivery_Drone_{d}_Region_{r}"
                )

        # Zero waste policy for package deliveries
        for r in range(n_regions):
            model.addCons(
                quicksum(route_vars[(u, f"region_{r}")] for u in G.predecessors(f"region_{r}")) == demand_levels[r],
                f"ZeroWastePolicy_Region_{r}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_drones': 600,
        'n_regions': 50,
        'min_operational_cost': 1000,
        'max_operational_cost': 2000,
        'min_delivery_cost': 800,
        'max_delivery_cost': 1500,
        'min_battery_limit': 1000,
        'max_battery_limit': 5000,
        'min_demand_level': 40.0,
        'max_demand_level': 50.0,
        'max_delivery_distance': 2400,
        'efficiency_factor': 10.5,
        'city_size': 2000,
    }

    drone_optimizer = DroneDeliveryOptimization(parameters, seed=seed)
    instance = drone_optimizer.generate_instance()
    solve_status, solve_time, objective_value = drone_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")