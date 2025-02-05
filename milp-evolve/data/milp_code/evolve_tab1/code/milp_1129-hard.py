import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DisasterResponseOptimization:
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

        neighborhood_revenue = {n: np.random.uniform(10, 100) for n in range(self.n_neighborhoods)}
        environmental_impact = {n: np.random.uniform(0, 1) for n in range(self.n_neighborhoods)}

        G = nx.erdos_renyi_graph(n=self.n_neighborhoods, p=self.route_prob, directed=True, seed=self.seed)
        for node in G.nodes:
            G.nodes[node]['households'] = np.random.randint(50, 500)
        for u, v in G.edges:
            G[u][v]['route_time'] = np.random.randint(5, 15)
            G[u][v]['route_capacity'] = np.random.randint(10, 30)
            G[u][v]['traffic_delay'] = np.random.uniform(1, self.traffic_delay_factor)
            G[u][v]['weather_sensitivity'] = np.random.uniform(0.5, 2.0)

        maintenance_costs = np.random.uniform(self.min_maintenance_cost, self.max_maintenance_cost, self.n_facilities)
        hazard_risk = np.random.uniform(0.1, 0.9, self.n_neighborhoods)

        # Generate mutual exclusivity pairs
        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            fac1 = random.randint(0, self.n_facilities - 1)
            fac2 = random.randint(0, self.n_facilities - 1)
            if fac1 != fac2:
                mutual_exclusivity_pairs.append((fac1, fac2))

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "neighborhood_revenue": neighborhood_revenue,
            "environmental_impact": environmental_impact,
            "G": G,
            "maintenance_costs": maintenance_costs,
            "hazard_risk": hazard_risk,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        neighborhood_revenue = instance['neighborhood_revenue']
        environmental_impact = instance['environmental_impact']
        G = instance['G']
        maintenance_costs = instance['maintenance_costs']
        hazard_risk = instance['hazard_risk']
        mutual_exclusivity_pairs = instance["mutual_exclusivity_pairs"]

        model = Model("DisasterResponseOptimization")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])

        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        route_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in G.edges}
        maintenance_vars = {f: model.addVar(vtype="C", name=f"Maintenance_{f}") for f in range(n_facilities)}
        environmental_impact_vars = {n: model.addVar(vtype="C", name=f"EnvImpact_{n}") for n in range(n_neighborhoods)}

        # Maintenance Cost Constraints
        for f in range(n_facilities):
            model.addCons(maintenance_vars[f] == self.maintenance_multiplier * facility_vars[f], f"MaintenanceCost_{f}")

        # Environmental Impact Constraints
        for n in range(n_neighborhoods):
            model.addCons(environmental_impact_vars[n] == environmental_impact[n] * quicksum(allocation_vars[f, n] for f in range(n_facilities)), f"EnvImpact_{n}")

        # Mutual Exclusivity Constraints
        for (fac1, fac2) in mutual_exclusivity_pairs:
            model.addCons(facility_vars[fac1] + facility_vars[fac2] <= 1, f"Exclusive_{fac1}_{fac2}")

        # Objective: Maximize total profitability while minimizing costs and environmental impacts
        model.setObjective(
            quicksum(neighborhood_revenue[n] * quicksum(allocation_vars[f, n] for f in range(n_facilities)) - 
                     environmental_impact_vars[n] for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(maintenance_costs[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)),
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

        # Route and flow constraints
        for u, v in G.edges:
            model.addCons(route_vars[u, v] <= G[u][v]['route_capacity'], f"RouteCapacity_{u}_{v}")

        # Flow conservation constraints with stochastic elements
        for node in G.nodes:
            model.addCons(
                quicksum(route_vars[u, node] for u in G.predecessors(node)) ==
                quicksum(route_vars[node, v] for v in G.successors(node)),
                f"FlowConservation_{node}"
            )

        # Budget constraint considering stochastic costs
        total_cost = quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) + quicksum(maintenance_costs[f] * maintenance_vars[f] for f in range(n_facilities))
        model.addCons(total_cost <= self.max_total_cost, "BudgetConstraint")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 60,
        'n_neighborhoods': 117,
        'min_fixed_cost': 157,
        'max_fixed_cost': 1970,
        'mean_transport_cost': 390,
        'stddev_transport_cost': 882,
        'min_transport_cost': 392,
        'max_transport_cost': 2133,
        'min_capacity': 260,
        'max_capacity': 1012,
        'route_prob': 0.31,
        'traffic_delay_factor': 486.0,
        'max_total_cost': 100000,
        'min_maintenance_cost': 2500,
        'max_maintenance_cost': 5000,
        'maintenance_multiplier': 13.5,
        'n_exclusive_pairs': 10,
    }

    optimizer = DisasterResponseOptimization(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")