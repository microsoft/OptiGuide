import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class SustainableHousingDevelopmentWithNetworkFlow:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# Data generation #################
    def generate_instance(self):
        assert self.min_value >= 0 and self.max_value >= self.min_value
        assert self.add_item_prob >= 0 and self.add_item_prob <= 1

        housing_values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_houses)
        project_bids = []

        while len(project_bids) < self.n_projects:
            bundle_size = np.random.randint(1, self.max_bundle_size + 1)
            bundle = np.random.choice(self.n_houses, size=bundle_size, replace=False)
            budget = max(housing_values[bundle].sum() + np.random.normal(0, 10), 0)
            complexity = np.random.poisson(lam=5)

            project_bids.append((bundle.tolist(), budget, complexity))

        bids_per_house = [[] for _ in range(self.n_houses)]
        for i, bid in enumerate(project_bids):
            bundle, budget, complexity = bid
            for house in bundle:
                bids_per_house[house].append(i)

        # Facility data generation
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        assignment_cost = np.random.normal(loc=5, scale=2, size=len(project_bids)).tolist()
        capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()
        maintenance_cost = np.random.lognormal(mean=3, sigma=1.0, size=n_facilities).tolist()

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            fac1 = random.randint(0, n_facilities - 1)
            fac2 = random.randint(0, n_facilities - 1)
            if fac1 != fac2:
                mutual_exclusivity_pairs.append((fac1, fac2))

        # Generate renewable energy usage rates for facilities
        renewable_usage = np.random.uniform(0.2, 1.0, size=n_facilities).tolist()

        # Generate resource availability for construction
        resource_availability = np.random.uniform(100, 300, size=self.n_houses).tolist()

        # Generate carbon offset data
        carbon_offset = np.random.normal(30, 5, size=n_facilities).tolist()

        # Generate labor wages
        labor_wages = np.random.uniform(15, 60, size=n_facilities).tolist()

        # Generate regulation compliance penalties
        compliance_penalties = np.random.uniform(0, 15, size=n_facilities).tolist()

        # Generate transportation idling costs
        transport_idle_cost = np.random.uniform(2, 12, size=n_facilities).tolist()
        sustainability_budget = np.random.randint(10000, 20000)

        # Generate seasonal demand factors for housing
        seasonal_demand = [random.randint(0, 1) for _ in range(self.n_routes)]
        transport_costs = np.random.uniform(10, 30, size=self.n_routes).tolist()

        # Generate minimum housing development levels
        min_housing_level = np.random.randint(80, 250, size=self.n_houses).tolist()

        # Generate a random network structure using NetworkX
        G = nx.erdos_renyi_graph(self.n_nodes, self.edge_probability, seed=self.seed)
        while nx.number_connected_components(G) > 1:
            G = nx.erdos_renyi_graph(self.n_nodes, self.edge_probability, seed=self.seed)

        network_edges = [(u, v) for u, v in G.edges()]

        # Generate renewable energy share data
        renewable_share = np.random.uniform(0.3, 0.9, size=self.n_routes).tolist()

        return {
            "project_bids": project_bids,
            "bids_per_house": bids_per_house,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "maintenance_cost": maintenance_cost,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "renewable_usage": renewable_usage,
            "resource_availability": resource_availability,
            "carbon_offset": carbon_offset,
            "labor_wages": labor_wages,
            "compliance_penalties": compliance_penalties,
            "transport_idle_cost": transport_idle_cost,
            "sustainability_budget": sustainability_budget,
            "seasonal_demand": seasonal_demand,
            "transport_costs": transport_costs,
            "min_housing_level": min_housing_level,
            "network_edges": network_edges,
            "renewable_share": renewable_share
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        project_bids = instance['project_bids']
        bids_per_house = instance['bids_per_house']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']
        setup_cost = instance['setup_cost']
        maintenance_cost = instance['maintenance_cost']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        renewable_usage = instance['renewable_usage']
        resource_availability = instance['resource_availability']
        carbon_offset = instance['carbon_offset']
        labor_wages = instance['labor_wages']
        compliance_penalties = instance['compliance_penalties']
        transport_idle_cost = instance['transport_idle_cost']
        sustainability_budget = instance['sustainability_budget']
        seasonal_demand = instance['seasonal_demand']
        transport_costs = instance['transport_costs']
        min_housing_level = instance['min_housing_level']
        network_edges = instance['network_edges']
        renewable_share = instance['renewable_share']

        model = Model("SustainableHousingDevelopmentWithNetworkFlow")

        bid_vars = {i: model.addVar(vtype="B", name=f"Bid_{i}") for i in range(len(project_bids))}
        y_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        x_vars = {(i, j): model.addVar(vtype="C", name=f"x_{i}_{j}", lb=0, ub=1) for i in range(len(project_bids)) for j in range(n_facilities)}
        facility_workload = {j: model.addVar(vtype="I", name=f"workload_{j}", lb=0) for j in range(n_facilities)}

        # New housing development variables
        housing_development_vars = {i: model.addVar(vtype="C", name=f"house_dev_{i}", lb=0) for i in range(self.n_houses)}

        # New energy usage variables
        energy_usage_vars = {j: model.addVar(vtype="C", name=f"energy_{j}", lb=0) for j in range(n_facilities)}

        # New resource usage variables
        resource_usage_vars = {i: model.addVar(vtype="C", name=f"resource_{i}", lb=0) for i in range(self.n_houses)}

        # New labor wages variables
        labor_wages_vars = {j: model.addVar(vtype="C", name=f"labor_wages_{j}", lb=0) for j in range(n_facilities)}

        # New carbon offset variables
        carbon_offset_vars = {j: model.addVar(vtype="C", name=f"carbon_offset_{j}", lb=0) for j in range(n_facilities)}

        # New transportation idling time variables
        transport_idle_time_vars = {j: model.addVar(vtype="C", name=f"idle_time_{j}", lb=0) for j in range(n_facilities)}

        # New seasonal demand factor variables
        seasonal_demand_vars = {r: demand_factor * instance['seasonal_demand'][r] for r, demand_factor in enumerate(instance['seasonal_demand'])}
        
        # New transport cost variables
        transport_cost_vars = {r: model.addVar(vtype="C", name=f"trans_cost_{r}", lb=10) for r in range(len(instance['transport_costs']))}

        # New minimum housing level variables
        housing_level_vars = {i: model.addVar(vtype="C", name=f"housing_{i}", lb=0) for i in range(self.n_houses)}

        # New network flow variables
        flow_vars = {edge: model.addVar(vtype="C", name=f"flow_{edge[0]}_{edge[1]}", lb=0, ub=self.max_flow_value) for edge in network_edges}

        # New renewable energy share variables
        renewable_share_vars = {r: model.addVar(vtype="C", name=f"renewable_{r}", lb=0) for r in range(len(renewable_share))}

        objective_expr = quicksum(budget * bid_vars[i] for i, (bundle, budget, complexity) in enumerate(project_bids)) \
                         - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(assignment_cost[i] * x_vars[i, j] for i in range(len(project_bids)) for j in range(n_facilities)) \
                         - quicksum(setup_cost[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(maintenance_cost[j] * facility_workload[j] for j in range(n_facilities)) \
                         - quicksum(complexity * bid_vars[i] for i, (bundle, budget, complexity) in enumerate(project_bids)) \
                         - quicksum(renewable_usage[j] * energy_usage_vars[j] * self.energy_cost for j in range(n_facilities)) \
                         - quicksum(labor_wages[j] * labor_wages_vars[j] for j in range(n_facilities)) \
                         - quicksum(carbon_offset[j] * carbon_offset_vars[j] for j in range(n_facilities)) \
                         - quicksum(compliance_penalties[j] * y_vars[j] for j in range(n_facilities)) \
                         - quicksum(transport_idle_cost[j] * transport_idle_time_vars[j] for j in range(n_facilities)) \
                         - quicksum(transport_cost_vars[r] * seasonal_demand_vars[r] for r in range(len(transport_cost_vars))) \
                         - quicksum(housing_level_vars[i] * min_housing_level[i] for i in range(self.n_houses)) \
                         - quicksum(renewable_share_vars[r] * renewable_share[r] for r in range(len(renewable_share)))

        # Constraints: Each house can only be part of one accepted bid
        for house, bid_indices in enumerate(bids_per_house):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"House_{house}")

        # Bid assignment to facility
        for i in range(len(project_bids)):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == bid_vars[i], f"BidFacility_{i}")

        # Facility capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(len(project_bids))) <= capacity[j] * y_vars[j], f"FacilityCapacity_{j}")

        # Facility workload constraints
        for j in range(n_facilities):
            model.addCons(facility_workload[j] == quicksum(x_vars[i, j] * project_bids[i][2] for i in range(len(project_bids))), f"Workload_{j}")

        # Energy usage constraints linked to workload
        for j in range(n_facilities):
            model.addCons(energy_usage_vars[j] == quicksum(x_vars[i, j] * renewable_usage[j] for i in range(len(project_bids))), f"EnergyUsage_{j}")

        # Resource usage constraints
        for i in range(self.n_houses):
            model.addCons(resource_usage_vars[i] <= resource_availability[i], f"Resource_{i}")

        # Labor wages constraints
        for j in range(n_facilities):
            model.addCons(labor_wages_vars[j] <= labor_wages[j], f"LaborWages_{j}")

        # Carbon offset constraints
        for j in range(n_facilities):
            model.addCons(carbon_offset_vars[j] <= carbon_offset[j], f"CarbonOffset_{j}")

        # Transportation idle time constraint
        model.addCons(quicksum(maintenance_cost[j] * transport_idle_time_vars[j] for j in range(n_facilities)) <= sustainability_budget, "SustainabilityBudget")

        # Further constraints to introduce complexity
        # Constraints on mutual exclusivity for facilities
        for fac1, fac2 in mutual_exclusivity_pairs:
            model.addCons(y_vars[fac1] + y_vars[fac2] <= 1, f"MutualExclusivity_{fac1}_{fac2}")

        # Housing development constraints
        for i in range(self.n_houses):
            model.addCons(housing_development_vars[i] >= min_housing_level[i], f"HousingDevelopment_{i}")

        # Transport cost constraints considering seasonal demand
        for r in range(len(transport_cost_vars)):
            model.addCons(transport_cost_vars[r] >= seasonal_demand[r] * transport_costs[r], f"TransportCost_{r}")

        # Network flow constraints
        for node in range(self.n_nodes):
            inflow = quicksum(flow_vars[edge] for edge in network_edges if edge[1] == node)
            outflow = quicksum(flow_vars[edge] for edge in network_edges if edge[0] == node)
            # Source and Sink nodes constraints
            if node == self.source_node:
                model.addCons(inflow == 0, f"SourceNode_{node}")
            elif node == self.sink_node:
                model.addCons(outflow == 0, f"SinkNode_{node}")
            else:
                model.addCons(inflow == outflow, f"FlowConservation_{node}")

        # Renewable energy share constraints
        for r in range(len(renewable_share)):
            model.addCons(renewable_share_vars[r] >= renewable_share[r], f"RenewableShare_{r}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_houses': 1875,
        'n_projects': 37,
        'min_value': 1125,
        'max_value': 6000,
        'max_bundle_size': 750,
        'add_item_prob': 0.31,
        'facility_min_count': 500,
        'facility_max_count': 1300,
        'complexity_mean': 975,
        'complexity_stddev': 700,
        'n_exclusive_pairs': 375,
        'energy_cost': 0.17,
        'n_routes': 1700,
        'n_nodes': 425,
        'edge_probability': 0.73,
        'max_flow_value': 450,
        'source_node': 0,
        'sink_node': 1698,
    }

    development = SustainableHousingDevelopmentWithNetworkFlow(parameters, seed=seed)
    instance = development.generate_instance()
    solve_status, solve_time = development.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")