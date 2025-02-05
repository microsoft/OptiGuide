import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_resources > 0 and self.n_zones > 0
        assert self.min_resource_cost >= 0 and self.max_resource_cost >= self.min_resource_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_resource_availability > 0 and self.max_resource_availability >= self.min_resource_availability

        resource_costs = np.random.randint(self.min_resource_cost, self.max_resource_cost + 1, self.n_resources)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_resources, self.n_zones))
        resource_availability = np.random.randint(self.min_resource_availability, self.max_resource_availability + 1, self.n_resources)
        zone_demands = np.random.randint(1, 10, self.n_zones)
        budget_limits = np.random.uniform(self.min_budget_limit, self.max_budget_limit, self.n_resources)
        distances = np.random.uniform(0, self.max_transport_distance, (self.n_resources, self.n_zones))
        vehicle_capacities = np.random.randint(5, 15, (self.n_resources, self.n_zones))
        penalty_costs = np.random.normal(self.penalty_cost_mean, self.penalty_cost_sd, self.n_zones)

        G = nx.DiGraph()
        node_pairs = []
        for r in range(self.n_resources):
            for z in range(self.n_zones):
                G.add_edge(f"resource_{r}", f"zone_{z}")
                node_pairs.append((f"resource_{r}", f"zone_{z}"))

        return {
            "resource_costs": resource_costs,
            "transport_costs": transport_costs,
            "resource_availability": resource_availability,
            "zone_demands": zone_demands,
            "budget_limits": budget_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "vehicle_capacities": vehicle_capacities,
            "penalty_costs": penalty_costs,
        }

    def solve(self, instance):
        resource_costs = instance['resource_costs']
        transport_costs = instance['transport_costs']
        resource_availability = instance['resource_availability']
        zone_demands = instance['zone_demands']
        budget_limits = instance['budget_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        vehicle_capacities = instance['vehicle_capacities']
        penalty_costs = instance['penalty_costs']

        model = Model("EmergencyResourceAllocation")
        n_resources = len(resource_costs)
        n_zones = len(transport_costs[0])

        # Decision variables
        resource_vars = {r: model.addVar(vtype="B", name=f"Resource_{r}") for r in range(n_resources)}
        allocation_vars = {(u, v): model.addVar(vtype="C", name=f"Allocation_{u}_{v}") for u, v in node_pairs}
        vehicle_vars = {(u, v): model.addVar(vtype="I", name=f"Vehicle_{u}_{v}") for u, v in node_pairs}
        zone_allocation = {f"alloc_{z + 1}": model.addVar(vtype="C", name=f"alloc_{z + 1}") for z in range(n_zones)}

        # Additional decision variables for unmet demand penalties
        unmet_demand_penalty_vars = {z: model.addVar(vtype="C", name=f"UnmetDemandPenalty_{z}") for z in range(n_zones)}

        # Objective: minimize the total cost including resource costs, transport costs, vehicle costs, and penalties for unmet demands.
        model.setObjective(
            quicksum(resource_costs[r] * resource_vars[r] for r in range(n_resources)) +
            quicksum(transport_costs[r, int(v.split('_')[1])] * allocation_vars[(u, v)] for (u, v) in node_pairs for r in range(n_resources) if u == f'resource_{r}') +
            quicksum(vehicle_vars[(u, v)] for (u, v) in node_pairs) +
            quicksum(penalty_costs[z] * unmet_demand_penalty_vars[z] for z in range(n_zones)),
            "minimize"
        )

        # Resource allocation constraint for each zone
        for z in range(n_zones):
            model.addCons(
                quicksum(allocation_vars[(u, f"zone_{z}")] for u in G.predecessors(f"zone_{z}")) == zone_demands[z], 
                f"Zone_{z}_NodeFlowConservation"
            )

        # Constraints: Zones can only receive resources if the resources are available
        for r in range(n_resources):
            for z in range(n_zones):
                model.addCons(
                    allocation_vars[(f"resource_{r}", f"zone_{z}")] <= budget_limits[r] * resource_vars[r], 
                    f"Resource_{r}_AllocationLimitByBudget_{z}"
                )

        # Constraints: Resources cannot exceed their availability
        for r in range(n_resources):
            model.addCons(
                quicksum(allocation_vars[(f"resource_{r}", f"zone_{z}")] for z in range(n_zones)) <= resource_availability[r], 
                f"Resource_{r}_MaxAvailability"
            )

        # Coverage constraint for zones
        for z in range(n_zones):
            model.addCons(
                quicksum(resource_vars[r] for r in range(n_resources) if distances[r, z] <= self.max_transport_distance) >= 1, 
                f"Zone_{z}_Coverage"
            )

        # Constraints: Vehicle capacities cannot be exceeded
        for u, v in node_pairs:
            model.addCons(vehicle_vars[(u, v)] <= vehicle_capacities[int(u.split('_')[1]), int(v.split('_')[1])], f"VehicleCapacity_{u}_{v}")

        # Constraints for unmet demand penalties
        for z in range(n_zones):
            model.addCons(unmet_demand_penalty_vars[z] >= zone_allocation[f"alloc_{z + 1}"] - zone_demands[z], f"UnmetDemandPenaltyConstraint_{z}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_resources': 524,
        'n_zones': 56,
        'min_transport_cost': 720,
        'max_transport_cost': 1518,
        'min_resource_cost': 1350,
        'max_resource_cost': 1579,
        'min_resource_availability': 228,
        'max_resource_availability': 2100,
        'min_budget_limit': 227,
        'max_budget_limit': 600,
        'max_transport_distance': 1239,
        'penalty_cost_mean': 787.5,
        'penalty_cost_sd': 56.25,
    }

    optimizer = EmergencyResourceAllocation(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")