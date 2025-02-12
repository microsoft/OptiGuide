import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class CommunityResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data Generation #################
    def generate_random_neighborhood(self):
        n_members = np.random.randint(self.min_n, self.max_n)
        neighborhood = nx.erdos_renyi_graph(n=n_members, p=self.er_prob, seed=self.seed)
        return neighborhood

    def generate_satisfaction_costs(self, neighborhood):
        for member in neighborhood.nodes:
            neighborhood.nodes[member]['satisfaction'] = np.random.randint(1, 100)
        for u, v in neighborhood.edges:
            neighborhood[u][v]['cost'] = np.random.normal((neighborhood.nodes[u]['satisfaction'] + neighborhood.nodes[v]['satisfaction']) / float(self.set_param), self.cost_sd)

    def generate_removable_roads(self, neighborhood):
        removable_roads = set()
        for road in neighborhood.edges:
            if np.random.random() <= self.road_removal_prob:
                removable_roads.add(road)
        return removable_roads

    def generate_service_demand(self, neighborhood):
        service_demands = {member: np.random.normal(self.demand_mean, self.demand_sd) for member in neighborhood.nodes}
        return service_demands

    def generate_instance(self):
        neighborhood = self.generate_random_neighborhood()
        self.generate_satisfaction_costs(neighborhood)
        removable_roads = self.generate_removable_roads(neighborhood)
        service_demand = self.generate_service_demand(neighborhood)

        zebra_resource_costs = {(u, v): np.random.randint(1, 10) for u, v in neighborhood.edges}
        carbon_constraints = {(u, v): np.random.randint(1, 10) for u, v in neighborhood.edges}
        holding_costs = {member: np.random.normal(self.holding_cost_mean, self.holding_cost_sd) for member in neighborhood.nodes}
        
        return {'neighborhood': neighborhood, 'removable_roads': removable_roads, 
                'service_demands': service_demand, 'zebra_resource_costs': zebra_resource_costs,
                'carbon_constraints': carbon_constraints, 'holding_costs': holding_costs}

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        neighborhood, removable_roads, service_demands, zebra_resource_costs, carbon_constraints, holding_costs = \
            instance['neighborhood'], instance['removable_roads'], instance['service_demands'], instance['zebra_resource_costs'], instance['carbon_constraints'], instance['holding_costs']

        model = Model("CommunityResourceAllocation")

        member_vars = {f"m{member}": model.addVar(vtype="B", name=f"m{member}") for member in neighborhood.nodes}
        road_vars = {f"r{u}_{v}": model.addVar(vtype="B", name=f"r{u}_{v}") for u, v in neighborhood.edges}
        service_vars = {f"s{u}_{v}": model.addVar(vtype="B", name=f"s{u}_{v}") for u, v in neighborhood.edges}

        # New variables for holding costs, service setup, zebra resource costs, carbon, and demand
        demand_level = {f"dem_{member}": model.addVar(vtype="C", name=f"dem_{member}") for member in neighborhood.nodes}
        service_setup = {f"setup_{u}_{v}": model.addVar(vtype="B", name=f"setup_{u}_{v}") for u, v in neighborhood.edges}

        # Objective function: Maximize satisfaction, minus costs
        objective_expr = quicksum(
            neighborhood.nodes[member]['satisfaction'] * member_vars[f"m{member}"]
            for member in neighborhood.nodes
        )

        objective_expr -= quicksum(
            neighborhood[u][v]['cost'] * road_vars[f"r{u}_{v}"]
            for u, v in removable_roads
        )

        objective_expr -= quicksum(
            holding_costs[member] * demand_level[f"dem_{member}"]
            for member in neighborhood.nodes
        )

        objective_expr -= quicksum(
            zebra_resource_costs[(u, v)] * service_setup[f"setup_{u}_{v}"]
            for u, v in neighborhood.edges
        )

        # Carbon constraints incorporated into the objective
        objective_expr -= quicksum(
            carbon_constraints[(u, v)] * road_vars[f"r{u}_{v}"]
            for u, v in neighborhood.edges
        )

        # Existing neighborhood constraints
        for u, v in neighborhood.edges:
            if (u, v) in removable_roads:
                model.addCons(
                    member_vars[f"m{u}"] + member_vars[f"m{v}"] - road_vars[f"r{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    member_vars[f"m{u}"] + member_vars[f"m{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )

        # Service setup constraints
        for u, v in neighborhood.edges:
            model.addCons(
                road_vars[f"r{u}_{v}"] >= service_setup[f"setup_{u}_{v}"],
                name=f"Setup_{u}_{v}"
            )
        
        # Demand constraints
        for member in neighborhood.nodes:
            model.addCons(
                demand_level[f"dem_{member}"] >= 0,
                name=f"Dem_{member}_nonneg"
            )
            model.addCons(
                demand_level[f"dem_{member}"] - service_demands[member] >= 0,
                name=f"Service_{member}_out"
            )
        
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 75,
        'max_n': 1000,
        'er_prob': 0.8,
        'set_type': 'SET1',
        'set_param': 1200.0,
        'road_removal_prob': 0.3,
        'cost_sd': 120.0,
        'holding_cost_mean': 100.0,
        'holding_cost_sd': 16.0,
        'demand_mean': 100.0,
        'demand_sd': 140.0,
    }

    cr_alloc = CommunityResourceAllocation(parameters, seed=seed)
    instance = cr_alloc.generate_instance()
    solve_status, solve_time = cr_alloc.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")