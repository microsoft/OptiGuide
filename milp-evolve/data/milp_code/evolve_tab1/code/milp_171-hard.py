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
            neighborhood[u][v]['cost_linear'] = np.random.normal(
                (neighborhood.nodes[u]['satisfaction'] + neighborhood.nodes[v]['satisfaction']) / float(self.set_param),
                self.cost_sd
            )

    def generate_removable_roads(self, neighborhood):
        removable_roads = set()
        for road in neighborhood.edges:
            if np.random.random() <= self.road_removal_prob:
                removable_roads.add(road)
        return removable_roads

    def generate_service_demand(self, neighborhood):
        service_demands = {member: np.random.normal(self.demand_mean, self.demand_sd) for member in neighborhood.nodes}
        return service_demands

    def generate_piecewise_costs(self):
        breakpoints = [20, 50, 80]
        slopes = [1, 2, 3, 4]  # Slopes for each segment
        return breakpoints, slopes

    def generate_time_windows(self, neighborhood):
        time_windows = {}
        for member in neighborhood.nodes:
            start_time = np.random.randint(0, 12)
            end_time = start_time + np.random.randint(1, 5)
            time_windows[member] = (start_time, end_time)
        return time_windows

    def generate_instance(self):
        neighborhood = self.generate_random_neighborhood()
        self.generate_satisfaction_costs(neighborhood)
        removable_roads = self.generate_removable_roads(neighborhood)
        service_demand = self.generate_service_demand(neighborhood)
        breakpoints, slopes = self.generate_piecewise_costs()
        time_windows = self.generate_time_windows(neighborhood)

        zebra_resource_costs = {(u, v): np.random.randint(1, 10) for u, v in neighborhood.edges}
        carbon_constraints = {(u, v): np.random.randint(1, 10) for u, v in neighborhood.edges}
        holding_costs = {member: np.random.normal(self.holding_cost_mean, self.holding_cost_sd) for member in neighborhood.nodes}
        
        return {
            'neighborhood': neighborhood, 
            'removable_roads': removable_roads, 
            'service_demands': service_demand, 
            'zebra_resource_costs': zebra_resource_costs, 
            'carbon_constraints': carbon_constraints, 
            'holding_costs': holding_costs, 
            'breakpoints': breakpoints, 
            'slopes': slopes,
            'time_windows': time_windows,
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        neighborhood, removable_roads, service_demands, zebra_resource_costs, carbon_constraints, holding_costs, breakpoints, slopes, time_windows = (
            instance['neighborhood'], 
            instance['removable_roads'], 
            instance['service_demands'], 
            instance['zebra_resource_costs'], 
            instance['carbon_constraints'], 
            instance['holding_costs'], 
            instance['breakpoints'], 
            instance['slopes'],
            instance['time_windows']
        )

        model = Model("CommunityResourceAllocation")

        member_vars = {f"m{member}": model.addVar(vtype="B", name=f"m{member}") for member in neighborhood.nodes}
        road_vars = {f"r{u}_{v}": model.addVar(vtype="B", name=f"r{u}_{v}") for u, v in neighborhood.edges}
        service_vars = {f"s{u}_{v}": model.addVar(vtype="B", name=f"s{u}_{v}") for u, v in neighborhood.edges}

        # New variables for holding costs, service setup, zebra resource costs, carbon, and demand
        demand_level = {f"dem_{member}": model.addVar(vtype="C", name=f"dem_{member}") for member in neighborhood.nodes}
        service_setup = {f"setup_{u}_{v}": model.addVar(vtype="B", name=f"setup_{u}_{v}") for u, v in neighborhood.edges}

        # Additional variables for piecewise linear cost
        piecewise_vars = {f"pcs_{u}_{v}_{k}": model.addVar(vtype="C", name=f"pcs_{u}_{v}_{k}") for u, v in neighborhood.edges for k in range(len(breakpoints)+1)}

        # New variables for time windows
        delivery_start_time = {f"start_{member}": model.addVar(vtype="C", name=f"start_{member}") for member in neighborhood.nodes}
        delivery_end_time = {f"end_{member}": model.addVar(vtype="C", name=f"end_{member}") for member in neighborhood.nodes}

        # Objective function: Maximize satisfaction, minus costs
        objective_expr = quicksum(
            neighborhood.nodes[member]['satisfaction'] * member_vars[f"m{member}"]
            for member in neighborhood.nodes
        )

        # Modify road cost constraint using piecewise linear functions
        for u, v in removable_roads:
            for k in range(len(breakpoints) + 1):
                if k == 0:
                    model.addCons(
                        piecewise_vars[f"pcs_{u}_{v}_{k}"] >= road_vars[f"r{u}_{v}"] * breakpoints[0],
                        name=f"Piecewise_cost_{u}_{v}_0"
                    )
                elif k == len(breakpoints):
                    model.addCons(
                        piecewise_vars[f"pcs_{u}_{v}_{k}"] >= road_vars[f"r{u}_{v}"] * max(neighborhood.nodes[u]['satisfaction'], neighborhood.nodes[v]['satisfaction']),
                        name=f"Piecewise_cost_{u}_{v}_{k}"
                    )
                else:
                    model.addCons(
                        piecewise_vars[f"pcs_{u}_{v}_{k}"] >= road_vars[f"r{u}_{v}"] * (breakpoints[k] - breakpoints[k-1]),
                        name=f"Piecewise_cost_{u}_{v}_{k}"
                    )
            objective_expr -= quicksum(
                slopes[k] * piecewise_vars[f"pcs_{u}_{v}_{k}"] for k in range(len(breakpoints) + 1)
            )

        # Holding costs
        objective_expr -= quicksum(
            holding_costs[member] * demand_level[f"dem_{member}"]
            for member in neighborhood.nodes
        )

        # Zebra resource costs
        objective_expr -= quicksum(
            zebra_resource_costs[(u, v)] * service_setup[f"setup_{u}_{v}"]
            for u, v in neighborhood.edges
        )

        # Carbon constraints
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

        # New constraints: Time windows and delivery duration
        for member in neighborhood.nodes:
            tw_start, tw_end = time_windows[member]
            model.addCons(
                delivery_start_time[f"start_{member}"] >= tw_start,
                name=f"TimeWindow_Start_{member}"
            )
            model.addCons(
                delivery_end_time[f"end_{member}"] <= tw_end,
                name=f"TimeWindow_End_{member}"
            )
            model.addCons(
                delivery_end_time[f"end_{member}"] - delivery_start_time[f"start_{member}"] >= 1,
                name=f"DeliveryDuration_{member}"
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
        'max_n': 750,
        'er_prob': 0.73,
        'set_type': 'SET1',
        'set_param': 2400.0,
        'road_removal_prob': 0.66,
        'cost_sd': 180.0,
        'holding_cost_mean': 350.0,
        'holding_cost_sd': 16.0,
        'demand_mean': 200.0,
        'demand_sd': 2100.0,
        'breakpoints': (30, 75, 120),
        'slopes': (2, 4, 6, 9),
        'time_window_start_sd': 0,
        'time_window_end_sd': 0,
    }

    cr_alloc = CommunityResourceAllocation(parameters, seed=seed)
    instance = cr_alloc.generate_instance()
    solve_status, solve_time = cr_alloc.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")