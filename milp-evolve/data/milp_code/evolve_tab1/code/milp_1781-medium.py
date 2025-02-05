import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class VaccineDistributionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_production_zones > 0 and self.n_distribution_zones > 0
        assert self.min_zone_cost >= 0 and self.max_zone_cost >= self.min_zone_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_production_capacity > 0 and self.max_production_capacity >= self.min_production_capacity
        assert self.max_transport_distance >= 0

        zone_costs = np.random.randint(self.min_zone_cost, self.max_zone_cost + 1, self.n_production_zones)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_production_zones, self.n_distribution_zones))
        production_capacities = np.random.randint(self.min_production_capacity, self.max_production_capacity + 1, self.n_production_zones)
        distribution_demands = np.random.randint(1, 10, self.n_distribution_zones)
        budget_limits = np.random.uniform(self.min_budget_limit, self.max_budget_limit, self.n_production_zones)
        distances = np.random.uniform(0, self.max_transport_distance, (self.n_production_zones, self.n_distribution_zones))
        shipment_costs = np.random.randint(1, 20, (self.n_production_zones, self.n_distribution_zones))
        shipment_capacities = np.random.randint(5, 15, (self.n_production_zones, self.n_distribution_zones))
        inventory_holding_costs = np.random.normal(self.holding_cost_mean, self.holding_cost_sd, self.n_distribution_zones)
        demand_levels = np.random.normal(self.demand_mean, self.demand_sd, self.n_distribution_zones)
        
        # Additional data for incorporating MKP
        weights = np.random.randint(self.min_weight, self.max_weight + 1, (self.n_production_zones, self.n_distribution_zones))
        profits = np.random.randint(self.min_profit, self.max_profit + 1, (self.n_production_zones, self.n_distribution_zones))
        knapsack_capacities = np.random.randint(self.min_knapsack_capacity, self.max_knapsack_capacity + 1, self.n_distribution_zones)

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.n_production_zones):
            for d in range(self.n_distribution_zones):
                G.add_edge(f"production_{p}", f"distribution_{d}")
                node_pairs.append((f"production_{p}", f"distribution_{d}"))

        # Data for Cargo Routing
        num_vehicles = random.randint(self.min_vehicles, self.max_vehicles)
        shipment_cost_vehicle = np.random.randint(100, 500, size=(self.n_distribution_zones, num_vehicles))
        operational_costs_vehicle = np.random.randint(2000, 5000, size=num_vehicles)
        vehicle_capacity = np.random.randint(10000, 20000, size=num_vehicles)
        break_points = np.sort(np.random.randint(0, self.max_capacity, size=(num_vehicles, self.num_break_points)))
        slopes = np.random.randint(5, 15, size=(num_vehicles, self.num_break_points + 1))

        return {
            "zone_costs": zone_costs,
            "transport_costs": transport_costs,
            "production_capacities": production_capacities,
            "distribution_demands": distribution_demands,
            "budget_limits": budget_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "shipment_costs": shipment_costs,
            "shipment_capacities": shipment_capacities,
            "inventory_holding_costs": inventory_holding_costs,
            "demand_levels": demand_levels,
            "weights": weights,
            "profits": profits,
            "knapsack_capacities": knapsack_capacities,
            "num_vehicles": num_vehicles,
            "shipment_cost_vehicle": shipment_cost_vehicle,
            "operational_costs_vehicle": operational_costs_vehicle,
            "vehicle_capacity": vehicle_capacity,
            "break_points": break_points,
            "slopes": slopes,
        }

    def solve(self, instance):
        zone_costs = instance['zone_costs']
        transport_costs = instance['transport_costs']
        production_capacities = instance['production_capacities']
        distribution_demands = instance['distribution_demands']
        budget_limits = instance['budget_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        shipment_costs = instance['shipment_costs']
        shipment_capacities = instance['shipment_capacities']
        inventory_holding_costs = instance['inventory_holding_costs']
        demand_levels = instance['demand_levels']
        weights = instance['weights']
        profits = instance['profits']
        knapsack_capacities = instance['knapsack_capacities']
        num_vehicles = instance['num_vehicles']
        shipment_cost_vehicle = instance['shipment_cost_vehicle']
        operational_costs_vehicle = instance['operational_costs_vehicle']
        vehicle_capacity = instance['vehicle_capacity']
        break_points = instance['break_points']
        slopes = instance['slopes']

        model = Model("VaccineDistributionOptimization")
        n_production_zones = len(zone_costs)
        n_distribution_zones = len(transport_costs[0])

        # Decision variables
        vaccine_vars = {p: model.addVar(vtype="B", name=f"Production_{p}") for p in range(n_production_zones)}
        shipment_vars = {(u, v): model.addVar(vtype="I", name=f"Shipment_{u}_{v}") for u, v in node_pairs}
        inventory_level = {f"inv_{d + 1}": model.addVar(vtype="C", name=f"inv_{d + 1}") for d in range(n_distribution_zones)}

        # Additional variables for vehicle routing and cargo
        Number_of_Vehicles = {v: model.addVar(vtype="B", name=f"Number_of_Vehicles_{v}") for v in range(num_vehicles)}
        Cargo_Shipment_Link = {(d, v): model.addVar(vtype="B", name=f"Cargo_Shipment_Link_{d}_{v}") for d in range(n_distribution_zones) for v in range(num_vehicles)}
        Maximum_Container_Usage = {v: model.addVar(vtype="C", name=f"Maximum_Container_Usage_{v}") for v in range(num_vehicles)}
        Heavy_Cargo_Limit = {(v, k): model.addVar(vtype="C", name=f"Heavy_Cargo_Limit_{v}_{k}") for v in range(num_vehicles) for k in range(len(break_points[0]) + 1)}

        # Objective
        objective_expr = (
            quicksum(zone_costs[p] * vaccine_vars[p] for p in range(n_production_zones)) +
            quicksum(transport_costs[p, int(v.split('_')[1])] * shipment_vars[(u, v)] for (u, v) in node_pairs for p in range(n_production_zones) if u == f'production_{p}') +
            quicksum(shipment_costs[int(u.split('_')[1]), int(v.split('_')[1])] * shipment_vars[(u, v)] for (u, v) in node_pairs) +
            quicksum(inventory_holding_costs[d] * inventory_level[f"inv_{d + 1}"] for d in range(n_distribution_zones)) +
            quicksum(Cargo_Shipment_Link[d, v] * shipment_cost_vehicle[d, v] for d in range(n_distribution_zones) for v in range(num_vehicles)) +
            quicksum(Number_of_Vehicles[v] * operational_costs_vehicle[v] for v in range(num_vehicles)) +
            quicksum(Heavy_Cargo_Limit[v, k] for v in range(num_vehicles) for k in range(len(break_points[v]) + 1))
        )

        # Additional objective to maximize profits from shipment
        profit_expr = quicksum(profits[int(u.split('_')[1]), int(v.split('_')[1])] * shipment_vars[(u, v)] for (u, v) in node_pairs)
        model.setObjective(objective_expr - profit_expr, "minimize")

        # Constraints
        for d in range(n_distribution_zones):
            model.addCons(
                quicksum(shipment_vars[(u, f"distribution_{d}")] for u in G.predecessors(f"distribution_{d}")) == distribution_demands[d], 
                f"Distribution_{d}_NodeFlowConservation"
            )

        for d in range(n_distribution_zones):
            model.addCons(
                quicksum(vaccine_vars[p] for p in range(n_production_zones) if distances[p, d] <= self.max_transport_distance) >= 1,
                f"Distribution_{d}_SetCovering"
            )

        for p in range(n_production_zones):
            for d in range(n_distribution_zones):
                model.addCons(
                    shipment_vars[(f"production_{p}", f"distribution_{d}")] <= budget_limits[p] * vaccine_vars[p],
                    f"Production_{p}_BudgetLimit_{d}"
                )

        for p in range(n_production_zones):
            model.addCons(
                quicksum(shipment_vars[(f"production_{p}", f"distribution_{d}")] for d in range(n_distribution_zones)) <= production_capacities[p],
                f"Production_{p}_MaxCapacity"
            )

        for u, v in node_pairs:
            model.addCons(shipment_vars[(u, v)] <= shipment_capacities[int(u.split('_')[1]), int(v.split('_')[1])], f"ShipmentCapacity_{u}_{v}")

        for d in range(n_distribution_zones):
            model.addCons(
                inventory_level[f"inv_{d + 1}"] - demand_levels[d] >= 0,
                f"Stock_{d + 1}_out"
            )

        for d in range(n_distribution_zones):
            model.addCons(
                quicksum(weights[int(u.split('_')[1]), d] * shipment_vars[(u, f"distribution_{d}")] for u in G.predecessors(f"distribution_{d}")) <= knapsack_capacities[d],
                f"KnapsackCapacity_{d}"
            )

        for d in range(n_distribution_zones):
            model.addCons(quicksum(Cargo_Shipment_Link[d, v] for v in range(num_vehicles)) == 1, name=f"destination_delivery_{d}")

        for v in range(num_vehicles):
            for d in range(n_distribution_zones):
                model.addCons(Cargo_Shipment_Link[d, v] <= Number_of_Vehicles[v], name=f"vehicle_shipment_link_{d}_{v}")

        for v in range(num_vehicles):
            model.addCons(quicksum(distribution_demands[d] * Cargo_Shipment_Link[d, v] for d in range(n_distribution_zones)) == Maximum_Container_Usage[v], name=f"maximum_container_usage_{v}")
            for k in range(len(break_points[v]) + 1):
                if k == 0:
                    model.addCons(Heavy_Cargo_Limit[v, k] >= slopes[v, k] * quicksum(distribution_demands[d] * Cargo_Shipment_Link[d, v] for d in range(n_distribution_zones)), name=f"heavy_cargo_limit_{v}_{k}")
                elif k == len(break_points[v]):
                    model.addCons(Heavy_Cargo_Limit[v, k] >= slopes[v, k] * (quicksum(distribution_demands[d] * Cargo_Shipment_Link[d, v] for d in range(n_distribution_zones)) - break_points[v, k-1]), name=f"heavy_cargo_limit_{v}_{k}")
                else:
                    model.addCons(Heavy_Cargo_Limit[v, k] >= slopes[v, k] * (quicksum(distribution_demands[d] * Cargo_Shipment_Link[d, v] for d in range(n_distribution_zones)) - break_points[v, k-1]) - Heavy_Cargo_Limit[v, k-1], name=f"heavy_cargo_limit_{v}_{k}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_production_zones': 9,
        'n_distribution_zones': 270,
        'min_transport_cost': 252,
        'max_transport_cost': 675,
        'min_zone_cost': 45,
        'max_zone_cost': 888,
        'min_production_capacity': 160,
        'max_production_capacity': 630,
        'min_budget_limit': 990,
        'max_budget_limit': 1800,
        'max_transport_distance': 1593,
        'holding_cost_mean': 2700.0,
        'holding_cost_sd': 62.5,
        'demand_mean': 1181.6,
        'demand_sd': 0.24,
        'min_weight': 7,
        'max_weight': 33,
        'min_profit': 30,
        'max_profit': 337,
        'min_knapsack_capacity': 250,
        'max_knapsack_capacity': 3000,
        'min_vehicles': 15,
        'max_vehicles': 15,
        'num_break_points': 36,
        'max_capacity': 20000
    }

    optimizer = VaccineDistributionOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")