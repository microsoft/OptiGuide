import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedFacilityLocation:
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
        transport_costs = np.abs(np.random.normal(self.mean_transport_cost, self.stddev_transport_cost, (self.n_facilities, self.n_neighborhoods)).astype(int))
        transport_costs = np.clip(transport_costs, self.min_transport_cost, self.max_transport_cost)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)

        financial_rewards = np.random.uniform(10, 100, self.n_neighborhoods)

        G = nx.erdos_renyi_graph(n=self.n_neighborhoods, p=self.route_prob, directed=True, seed=self.seed)
        for node in G.nodes:
            G.nodes[node]['households'] = np.random.randint(50, 500)
        for u, v in G.edges:
            G[u][v]['route_time'] = np.random.randint(5, 15)
            G[u][v]['route_capacity'] = np.random.randint(10, 30)

        high_priority_nodes = random.sample(G.nodes, len(G.nodes) // 3)
        patrol_times = {i: np.random.randint(1, 10, size=self.n_facilities).tolist() for i in high_priority_nodes}

        # Example of stochastic demand
        stochastic_demands = {n: np.random.normal(loc=100, scale=20) for n in range(self.n_neighborhoods)}

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "financial_rewards": financial_rewards,
            "G": G,
            "high_priority_nodes": high_priority_nodes,
            "patrol_times": patrol_times,
            "stochastic_demands": stochastic_demands,
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        financial_rewards = instance['financial_rewards']
        G = instance['G']
        high_priority_nodes = instance['high_priority_nodes']
        patrol_times = instance['patrol_times']
        stochastic_demands = instance['stochastic_demands']

        model = Model("EnhancedFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])

        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        z_vars = {(f, n): model.addVar(vtype="B", name=f"ConvexHull_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        route_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in G.edges}

        patrol_vars = {(node, j): model.addVar(vtype="B", name=f"Patrol_{node}_{j}") for node in high_priority_nodes for j in range(n_facilities)}

        energy_vars = {f: model.addVar(vtype="C", name=f"Energy_{f}") for f in range(n_facilities)}
        carbon_vars = {(f, n): model.addVar(vtype="C", name=f"Carbon_{f}_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}

        ### Modified Constraint: Convex Hull Formulation for Allocation Constraints
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] + z_vars[f, n] <= 1, f"ConvexHullConstraint1_{f}_{n}")
                model.addCons(facility_vars[f] + z_vars[f, n] >= 1, f"ConvexHullConstraint2_{f}_{n}")

        ### Energy Consumption Constraints
        for f in range(n_facilities):
            model.addCons(energy_vars[f] == quicksum(allocation_vars[f, n] * self.energy_per_neighborhood for n in range(n_neighborhoods)), f"EnergyConsumption_{f}")

        ### Carbon Emissions Constraints
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(carbon_vars[f, n] == allocation_vars[f, n] * transport_costs[f][n] * self.emission_per_cost, f"CarbonEmission_{f}_{n}")

        ### Robust constraints for stochastic demand
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] * stochastic_demands[n] for f in range(n_facilities)) <= self.demand_upper_bound, f"RobustDemand_{n}")

        model.setObjective(
            quicksum(financial_rewards[n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(G[u][v]['route_time'] * route_vars[u, v] for u, v in G.edges) -
            quicksum(self.patrol_fees * patrol_vars[node, j] for j in range(n_facilities) for node in high_priority_nodes) -
            quicksum(energy_vars[f] * self.energy_cost for f in range(n_facilities)) -
            quicksum(carbon_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)),
            "maximize"
        )

        # Constraints: Each neighborhood is served by exactly one facility
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")

        # Constraints: Only open facilities can serve neighborhoods
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Service_{n}")

        # Constraints: Facilities cannot exceed their capacity
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_Capacity")

        # Constraints: Minimum service level for facilities
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) >= self.min_service_level * facility_vars[f], f"Facility_{f}_MinService")

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

        # Patrol constraints for high priority nodes
        for node in high_priority_nodes:
            model.addCons(
                quicksum(patrol_vars[node, j] for j in range(n_facilities)) >= 1,
                f"PatrolCoverage_{node}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 27,
        'n_neighborhoods': 210,
        'min_fixed_cost': 1890,
        'max_fixed_cost': 2106,
        'mean_transport_cost': 393,
        'stddev_transport_cost': 2250,
        'min_transport_cost': 379,
        'max_transport_cost': 2250,
        'min_capacity': 1392,
        'max_capacity': 2025,
        'min_service_level': 0,
        'route_prob': 0.59,
        'patrol_fees': 3000,
        'energy_per_neighborhood': 750,
        'energy_cost': 0.73,
        'emission_per_cost': 0.73,
        'demand_upper_bound': 2250,
    }

    location_optimizer = EnhancedFacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")