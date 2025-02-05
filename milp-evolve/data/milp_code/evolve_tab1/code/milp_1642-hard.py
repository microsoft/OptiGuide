import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict
from scipy.spatial import Delaunay

class EmergencyResponseOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_vehicles > 0 and self.n_zones > 0
        assert self.min_operational_cost >= 0 and self.max_operational_cost >= self.min_operational_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_power_capacity > 0 and self.max_power_capacity >= self.min_power_capacity

        operational_costs = np.random.randint(self.min_operational_cost, self.max_operational_cost + 1, self.n_vehicles)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_vehicles, self.n_zones))
        fuel_costs = np.random.uniform(self.min_fuel_cost, self.max_fuel_cost, (self.n_vehicles, self.n_zones))
        delay_factors = np.random.uniform(self.min_delay_factor, self.max_delay_factor, self.n_zones)
        
        power_capacities = np.random.randint(self.min_power_capacity, self.max_power_capacity + 1, self.n_vehicles)
        zone_demands = np.random.randint(100, 1000, self.n_zones)
        service_requirements = np.random.uniform(self.min_service_level, self.max_service_level, self.n_zones)
        distances = np.random.uniform(0, self.max_service_distance, (self.n_vehicles, self.n_zones))
        
        zone_positions = np.random.rand(self.n_zones, 2) * self.city_size
        vehicle_positions = np.random.rand(self.n_vehicles, 2) * self.city_size

        G = nx.DiGraph()
        node_pairs = []
        for m in range(self.n_vehicles):
            for n in range(self.n_zones):
                G.add_edge(f"vehicle_{m}", f"zone_{n}")
                node_pairs.append((f"vehicle_{m}", f"zone_{n}"))

        # Generate generator data
        generator_costs = np.random.randint(self.min_generator_cost, self.max_generator_cost + 1, self.n_generators)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_generators, self.n_vehicles))
        generator_capacities = np.random.randint(self.min_generator_capacity, self.max_generator_capacity + 1, self.n_generators)

        return {
            "operational_costs": operational_costs,
            "transport_costs": transport_costs,
            "fuel_costs": fuel_costs,
            "delay_factors": delay_factors,
            "power_capacities": power_capacities,
            "zone_demands": zone_demands,
            "service_requirements": service_requirements,
            "distances": distances,
            "zone_positions": zone_positions,
            "vehicle_positions": vehicle_positions,
            "graph": G,
            "node_pairs": node_pairs,
            "generator_costs": generator_costs,
            "delivery_costs": delivery_costs,
            "generator_capacities": generator_capacities
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        transport_costs = instance['transport_costs']
        fuel_costs = instance['fuel_costs']
        delay_factors = instance['delay_factors']
        power_capacities = instance['power_capacities']
        zone_demands = instance['zone_demands']
        service_requirements = instance['service_requirements']
        distances = instance['distances']
        zone_positions = instance['zone_positions']
        vehicle_positions = instance['vehicle_positions']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        generator_costs = instance['generator_costs']
        delivery_costs = instance['delivery_costs']
        generator_capacities = instance['generator_capacities']
        
        model = Model("EmergencyResponseOptimization")
        n_vehicles = len(operational_costs)
        n_zones = len(transport_costs[0])
        n_generators = len(generator_costs)

        # Decision variables
        open_vars = {m: model.addVar(vtype="B", name=f"Vehicle_{m}") for m in range(n_vehicles)}
        schedule_vars = {(u, v): model.addVar(vtype="C", name=f"Schedule_{u}_{v}") for u, v in node_pairs}
        coverage_vars = {n: model.addVar(vtype="B", name=f"Coverage_Zone_{n}") for n in range(n_zones)}
        generator_vars = {g: model.addVar(vtype="B", name=f"Generator_{g}") for g in range(n_generators)}
        delivery_vars = {(g, m): model.addVar(vtype="C", name=f"Delivery_{g}_{m}") for g in range(n_generators) for m in range(n_vehicles)}
        
        electric_vars = {m: model.addVar(vtype="B", name=f"Electric_{m}") for m in range(n_vehicles)}
        hybrid_vars = {m: model.addVar(vtype="B", name=f"Hybrid_{m}") for m in range(n_vehicles)}
        combustion_vars = {m: model.addVar(vtype="B", name=f"Combustion_{m}") for m in range(n_vehicles)}

        # Objective: minimize the total cost including operational costs, transport costs, generator costs, and fuel costs.
        model.setObjective(
            quicksum(operational_costs[m] * open_vars[m] for m in range(n_vehicles)) +
            quicksum(transport_costs[m, int(v.split('_')[1])] * schedule_vars[(u, v)] for (u, v) in node_pairs for m in range(n_vehicles) if u == f'vehicle_{m}') +
            quicksum(fuel_costs[m, int(v.split('_')[1])] * schedule_vars[(u, v)] for (u, v) in node_pairs for m in range(n_vehicles) if u == f'vehicle_{m}') +
            quicksum(generator_costs[g] * generator_vars[g] for g in range(n_generators)) +
            quicksum(delivery_costs[g, m] * delivery_vars[g, m] for g in range(n_generators) for m in range(n_vehicles)),
            "minimize"
        )

        # Convex Hull Coverage Constraint
        tri = Delaunay(vehicle_positions)
        for n in range(n_zones):
            neighborhood_pos = zone_positions[n]
            hull_condition = tri.find_simplex(neighborhood_pos) >= 0
            model.addCons(
                coverage_vars[n] == hull_condition,
                f"ConvexHullCoverage_Zone_{n}"
            )

        # Power capacity limits for each vehicle
        for m in range(n_vehicles):
            model.addCons(
                quicksum(schedule_vars[(f"vehicle_{m}", f"zone_{n}")] for n in range(n_zones)) <= power_capacities[m], 
                f"LinkedPowerCapacity_Vehicle_{m}"
            )

        # Minimum service coverage for each zone
        for n in range(n_zones):
            model.addCons(
                quicksum(schedule_vars[(u, f"zone_{n}")] for u in G.predecessors(f"zone_{n}")) >= service_requirements[n], 
                f"MinimumServiceCoverage_Zone_{n}"
            )

        # Efficient scheduling for vehicles
        for m in range(n_vehicles):
            for n in range(n_zones):
                model.addCons(
                    schedule_vars[(f"vehicle_{m}", f"zone_{n}")] <= self.efficient_schedule_factor * open_vars[m], 
                    f"EfficientSchedulingParameters_Vehicle_{m}_Zone_{n}"
                )

        # Additional load distribution costs
        for m in range(n_vehicles):
            model.addCons(
                quicksum(schedule_vars[(f"vehicle_{m}", f"zone_{n}")] for n in range(n_zones)) <= self.heavy_load_factor, 
                f"HeavyLoadDistributionCosts_Vehicle_{m}"
            )

        # Only active generators can deliver to vehicles
        for g in range(n_generators):
            for m in range(n_vehicles):
                model.addCons(
                    delivery_vars[g, m] <= generator_capacities[g] * generator_vars[g],
                    f"Generator_{g}_Service_Vehicle_{m}"
                )
        
        # Generator capacity limits
        for g in range(n_generators):
            model.addCons(
                quicksum(delivery_vars[g, m] for m in range(n_vehicles)) <= generator_capacities[g],
                f"GeneratorCapacity_{g}"
            )
        
        # Vehicles must be replenished from generators
        for m in range(n_vehicles):
            model.addCons(
                quicksum(delivery_vars[g, m] for g in range(n_generators)) >= quicksum(schedule_vars[(f"vehicle_{m}", f"zone_{n}")] for n in range(n_zones)),
                f"VehicleReplenishment_{m}"
            )
        
        # Ensure at least 40% of vehicles are electric or hybrid
        model.addCons(
            quicksum(electric_vars[m] + hybrid_vars[m] for m in range(n_vehicles)) >= 0.4 * n_vehicles,
            f"FleetComposition"
        )
        
        # At most one type per vehicle
        for m in range(n_vehicles):
            model.addCons(
                electric_vars[m] + hybrid_vars[m] + combustion_vars[m] == 1,
                f"SingleTypePerVehicle_{m}"
            )

        # Penalize delivery delays due to traffic in each zone
        for n in range(n_zones):
            delay_factor = delay_factors[n]
            model.addCons(
                quicksum(schedule_vars[(u, f"zone_{n}")] * delay_factor for u in G.predecessors(f"zone_{n}")) <= zone_demands[n] * (1 + delay_factor),
                f"DeliveryDelayPenalty_Zone_{n}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_vehicles': 50,
        'n_zones': 450,
        'min_transport_cost': 50,
        'max_transport_cost': 250,
        'min_operational_cost': 1500,
        'max_operational_cost': 1500,
        'min_power_capacity': 250,
        'max_power_capacity': 1875,
        'min_service_level': 7,
        'max_service_level': 10,
        'max_service_distance': 2000,
        'efficient_schedule_factor': 50.0,
        'heavy_load_factor': 2000,
        'city_size': 4000,
        'n_generators': 40,
        'min_generator_cost': 100,
        'max_generator_cost': 7000,
        'min_delivery_cost': 10,
        'max_delivery_cost': 2000,
        'min_generator_capacity': 2000,
        'max_generator_capacity': 4000,
        'min_fuel_cost': 0.45,
        'max_fuel_cost': 13.5,
        'min_delay_factor': 0.59,
        'max_delay_factor': 0.8,
    }

    emergency_optimizer = EmergencyResponseOptimization(parameters, seed=seed)
    instance = emergency_optimizer.generate_instance()
    solve_status, solve_time, objective_value = emergency_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")