import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class PackageDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_neighborhoods >= self.n_facilities
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.normal(self.mean_transport_cost, self.stddev_transport_cost, (self.n_facilities, self.n_neighborhoods)).astype(int)
        transport_costs = np.clip(transport_costs, self.min_transport_cost, self.max_transport_cost)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)

        delivery_rewards = np.random.uniform(10, 100, self.n_neighborhoods)

        G = nx.erdos_renyi_graph(n=self.n_neighborhoods, p=self.route_prob, directed=True, seed=self.seed)
        for node in G.nodes:
            G.nodes[node]['households'] = np.random.randint(50, 500)
        for u, v in G.edges:
            G[u][v]['route_time'] = np.random.randint(5, 15)
            G[u][v]['route_capacity'] = np.random.randint(10, 30)

        delivery_frequencies = np.random.randint(1, 5, self.n_neighborhoods)
        
        vehicle_types = ['van', 'truck', 'drone']
        vehicle_costs = {'van': 500, 'truck': 1000, 'drone': 2000}
        skill_levels = ['basic', 'intermediate', 'advanced']

        packaging_materials = np.random.choice(['recyclable', 'biodegradable'], self.n_facilities, p=[0.5, 0.5])   

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "delivery_rewards": delivery_rewards,
            "G": G,
            "delivery_frequencies": delivery_frequencies,
            "vehicle_types": vehicle_types,
            "vehicle_costs": vehicle_costs,
            "skills_levels": skill_levels,
            "packaging_materials": packaging_materials
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        delivery_rewards = instance['delivery_rewards']
        G = instance['G']
        delivery_frequencies = instance['delivery_frequencies']
        vehicle_types = instance['vehicle_types']
        vehicle_costs = instance['vehicle_costs']
        skills_levels = instance['skills_levels']
        packaging_materials = instance['packaging_materials']

        model = Model("PackageDeliveryOptimization")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])

        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        route_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in G.edges}
        energy_vars = {f: model.addVar(vtype="C", name=f"Energy_{f}") for f in range(n_facilities)}
        vehicle_allocation_vars = {(f, v): model.addVar(vtype="B", name=f"Facility_{f}_Vehicle_{v}") for f in range(n_facilities) for v in vehicle_types}
        skill_allocation_vars = {(f, s): model.addVar(vtype="B", name=f"Facility_{f}_Skill_{s}") for f in range(n_facilities) for s in skills_levels}

        # Energy Consumption Constraints
        for f in range(n_facilities):
            model.addCons(energy_vars[f] == quicksum(allocation_vars[f, n] * self.energy_per_neighborhood for n in range(n_neighborhoods)), f"EnergyConsumption_{f}")

        # Modified Optimization Objective 
        model.setObjective(
            quicksum(delivery_rewards[n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(route_vars[u, v] * G[u][v]['route_time'] for u, v in G.edges) -
            quicksum(energy_vars[f] * self.energy_cost for f in range(n_facilities)) -
            quicksum(vehicle_costs[v] * vehicle_allocation_vars[f, v] for f in range(n_facilities) for v in vehicle_types),
            "maximize"
        )

        # Constraints: Set Packing constraint, each neighborhood is served by exactly one vehicle
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Delivery")

        # Constraints: Only deliverable by opened facility, matching original model
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Delivery_{n}")

        # Constraints: Facility cannot exceed vehicle capacity
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_VehicleCapacity")

        # Constraints: Minimum level presence for each neighborhoods, based on delivery frequencies
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) >= self.min_service_level * facility_vars[f] * delivery_frequencies.mean(), f"Facility_{f}_MinDelivery")

        # Route and flow constraints
        for u, v in G.edges:
            model.addCons(route_vars[u, v] <= G[u][v]['route_capacity'], f"RouteCapacity_{u}_{v}")

        # Flow conservation constraints
        for node in G.nodes:
            model.addCons(
                quicksum(route_vars[u, node] for u in G.predecessors(node)) ==
                quicksum(route_vars[node, v] for v in G.successors(node)),
                f"FlowConservation_{node}"
            )
        
        # Budget constraint
        total_operational_cost = quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) + quicksum(vehicle_costs[v] * vehicle_allocation_vars[f, v] for f in range(n_facilities) for v in vehicle_types)
        model.addCons(total_operational_cost <= self.max_operational_cost, "BudgetConstraint")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 45,
        'n_neighborhoods': 117,
        'min_fixed_cost': 26,
        'max_fixed_cost': 492,
        'mean_transport_cost': 58,
        'stddev_transport_cost': 63,
        'min_transport_cost': 28,
        'max_transport_cost': 711,
        'min_capacity': 104,
        'max_capacity': 1138,
        'min_service_level': 0.66,
        'route_prob': 0.59,
        'energy_per_neighborhood': 84,
        'energy_cost': 0.66,
        'max_operational_cost': 100000,
    }
    
    delivery_optimizer = PackageDeliveryOptimization(parameters, seed=42)
    instance = delivery_optimizer.generate_instance()
    solve_status, solve_time, objective_value = delivery_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")