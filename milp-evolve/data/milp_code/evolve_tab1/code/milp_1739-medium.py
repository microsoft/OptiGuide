import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FleetManagementOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_vehicles > 0 and self.n_delivery_zones > 0
        assert self.min_vehicle_cost >= 0 and self.max_vehicle_cost >= self.min_vehicle_cost
        assert self.min_fuel_cost >= 0 and self.max_fuel_cost >= self.min_fuel_cost
        assert self.min_vehicle_capacity > 0 and self.max_vehicle_capacity >= self.min_vehicle_capacity
        assert self.max_emissions_limit > 0

        vehicle_costs = np.random.randint(self.min_vehicle_cost, self.max_vehicle_cost + 1, self.n_vehicles)
        fuel_costs = np.random.randint(self.min_fuel_cost, self.max_fuel_cost + 1, (self.n_vehicles, self.n_delivery_zones))
        vehicle_capacities = np.random.randint(self.min_vehicle_capacity, self.max_vehicle_capacity + 1, self.n_vehicles)
        delivery_demands = np.random.randint(1, self.max_delivery_demand + 1, self.n_delivery_zones)
        budget_constraints = np.random.uniform(self.min_budget_constraint, self.max_budget_constraint, self.n_vehicles)
        emission_factors = np.random.uniform(0, self.max_emission_factor, (self.n_vehicles, self.n_delivery_zones))
        energy_limits = np.random.uniform(self.min_energy_limit, self.max_energy_limit, self.n_vehicles)

        G = nx.DiGraph()
        route_pairs = []
        for v in range(self.n_vehicles):
            for d in range(self.n_delivery_zones):
                G.add_edge(f"vehicle_{v}", f"delivery_{d}")
                route_pairs.append((f"vehicle_{v}", f"delivery_{d}"))

        return {
            "vehicle_costs": vehicle_costs,
            "fuel_costs": fuel_costs,
            "vehicle_capacities": vehicle_capacities,
            "delivery_demands": delivery_demands,
            "budget_constraints": budget_constraints,
            "emission_factors": emission_factors,
            "energy_limits": energy_limits,
            "graph": G,
            "route_pairs": route_pairs,
        }

    def solve(self, instance):
        vehicle_costs = instance['vehicle_costs']
        fuel_costs = instance['fuel_costs']
        vehicle_capacities = instance['vehicle_capacities']
        delivery_demands = instance['delivery_demands']
        budget_constraints = instance['budget_constraints']
        emission_factors = instance['emission_factors']
        energy_limits = instance['energy_limits']
        G = instance['graph']
        route_pairs = instance['route_pairs']

        model = Model("FleetManagementOptimization")
        n_vehicles = len(vehicle_costs)
        n_delivery_zones = len(fuel_costs[0])

        # Decision variables
        equipment_usage = {v: model.addVar(vtype="B", name=f"EquipmentUsage_{v}") for v in range(n_vehicles)}
        vehicle_load = {(u, v): model.addVar(vtype="C", name=f"VehicleLoad_{u}_{v}") for u, v in route_pairs}

        # Objective: minimize the total cost including vehicle operational costs, fuel costs, and emission penalties.
        model.setObjective(
            quicksum(vehicle_costs[v] * equipment_usage[v] for v in range(n_vehicles)) +
            quicksum(fuel_costs[v, int(d.split('_')[1])] * vehicle_load[(u, d)] for (u, d) in route_pairs for v in range(n_vehicles) if u == f'vehicle_{v}') +
            quicksum(emission_factors[v, int(d.split('_')[1])] * vehicle_load[(u, d)] for (u, d) in route_pairs for v in range(n_vehicles) if u == f'vehicle_{v}'),
            "minimize"
        )

        # Flow conservation constraints
        for v in range(n_vehicles):
            model.addCons(
                quicksum(vehicle_load[(f"vehicle_{v}", f"delivery_{d}")] for d in range(n_delivery_zones)) <= vehicle_capacities[v] * equipment_usage[v],
                f"Vehicle_{v}_LoadCapacity"
            )
            model.addCons(
                quicksum(vehicle_load[(f"vehicle_{v}", f"delivery_{d}")] for d in range(n_delivery_zones)) <= budget_constraints[v],
                f"Vehicle_{v}_BudgetConstraint"
            )
            model.addCons(
                quicksum(emission_factors[v, d] * vehicle_load[(f"vehicle_{v}", f"delivery_{d}")] for d in range(n_delivery_zones)) <= energy_limits[v],
                f"Vehicle_{v}_EnergyLimit"
            )

        for d in range(n_delivery_zones):
            model.addCons(
                quicksum(vehicle_load[(f"vehicle_{v}", f"delivery_{d}")] for v in range(n_vehicles)) == delivery_demands[d],
                f"Delivery_{d}_LoadBalance"
            )

        # Total emissions constraint
        model.addCons(
            quicksum(emission_factors[v, int(d.split('_')[1])] * vehicle_load[(u, d)] for (u, d) in route_pairs for v in range(n_vehicles) if u == f'vehicle_{v}') <= self.max_emissions_limit,
            "Total_Emissions_Limit"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_vehicles': 150,
        'n_delivery_zones': 60,
        'min_fuel_cost': 400,
        'max_fuel_cost': 2000,
        'min_vehicle_cost': 3000,
        'max_vehicle_cost': 5000,
        'min_vehicle_capacity': 40,
        'max_vehicle_capacity': 1000,
        'min_budget_constraint': 400,
        'max_budget_constraint': 3000,
        'max_emission_factor': 70,
        'min_energy_limit': 100,
        'max_energy_limit': 200,
        'max_delivery_demand': 60,
        'max_emissions_limit': 5000,
    }

    optimizer = FleetManagementOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")