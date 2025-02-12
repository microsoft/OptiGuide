import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum


class WarehouseLayoutOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_units > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_facility_space > 0 and self.max_facility_space >= self.min_facility_space

        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_units))
        spaces = np.random.randint(self.min_facility_space, self.max_facility_space + 1, self.n_facilities)
        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_units).astype(int)

        G = nx.erdos_renyi_graph(self.n_facilities, self.graph_density, seed=self.seed)
        travel_times = {(u, v): np.random.randint(10, 60) for u, v in G.edges}
        travel_costs = {(u, v): np.random.uniform(10.0, 60.0) for u, v in G.edges}

        # Generate new data for social and environmental impacts
        social_impact = np.random.randint(self.min_social_impact, self.max_social_impact + 1, self.n_facilities)
        carbon_emissions = np.random.uniform(self.min_carbon_emission, self.max_carbon_emission, (self.n_facilities, self.n_units))

        return {
            "facility_costs": facility_costs,
            "transport_costs": transport_costs,
            "spaces": spaces,
            "demands": demands,
            "graph": G,
            "travel_times": travel_times,
            "travel_costs": travel_costs,
            "social_impact": social_impact,
            "carbon_emissions": carbon_emissions,
        }

    # MILP modeling
    def solve(self, instance):
        facility_costs = instance["facility_costs"]
        transport_costs = instance["transport_costs"]
        spaces = instance["spaces"]
        demands = instance["demands"]
        G = instance["graph"]
        travel_times = instance["travel_times"]
        travel_costs = instance["travel_costs"]
        social_impact = instance["social_impact"]
        carbon_emissions = instance["carbon_emissions"]

        model = Model("WarehouseLayoutOptimization")
        n_facilities = len(facility_costs)
        n_units = len(transport_costs[0])

        BigM = self.bigM

        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        route_vars = {(u, v): model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        aux_vars = {(f, u): model.addVar(vtype="C", lb=0, name=f"Aux_{f}_{u}") for f in range(n_facilities) for u in range(n_units)}

        # New decision variables
        community_disruption_vars = {f: model.addVar(vtype="B", name=f"CommunityDisruption_{f}_LocalImpact") for f in range(n_facilities)}
        carbon_emission_vars = {(f, u): model.addVar(vtype="C", lb=0, name=f"CarbonEmission_{f}_{u}") for f in range(n_facilities) for u in range(n_units)}

        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f, u] * transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)) +
            quicksum(travel_costs[(u, v)] * route_vars[(u, v)] for u, v in G.edges) +
            quicksum(self.social_impact_weight * social_impact[f] * community_disruption_vars[f] for f in range(n_facilities)) + 
            quicksum(self.carbon_emission_weight * carbon_emissions[f, u] * carbon_emission_vars[f, u] for f in range(n_facilities) for u in range(n_units)), 
            "minimize"
        )

        # Constraints: Each unit demand should be met by at least one facility
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) >= 1, f"Unit_{u}_Demand")

        # Constraints: Only open facilities can serve units
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f], f"Facility_{f}_Serve_{u}")

        # Constraints: Facilities cannot exceed their space
        for f in range(n_facilities):
            model.addCons(quicksum(demands[u] * transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")

        # Constraints: Only open facilities can route edges
        for u, v in G.edges:
            model.addCons(route_vars[(u, v)] <= facility_vars[u], f"Route_{u}_{v}_facility_{u}")
            model.addCons(route_vars[(u, v)] <= facility_vars[v], f"Route_{u}_{v}_facility_{v}")

        # New constraints using Big M formulation for more complex logic
        for f in range(n_facilities):
            for u in range(n_units):
                # Auxiliary variables to ensure conditional logic is enforced ###
                model.addCons(aux_vars[(f, u)] >= transport_vars[f, u] - BigM * (1 - facility_vars[f]), f"bigM_aux1_{f}_{u}")
                model.addCons(aux_vars[(f, u)] <= transport_vars[f, u], f"bigM_aux2_{f}_{u}")

        # Ensure social and environmental impacts are considered
        for f in range(n_facilities):
            model.addCons(community_disruption_vars[f] == facility_vars[f], f"Community_Disruption_{f}")

        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(carbon_emission_vars[(f, u)] >= demands[u] * self.transport_distance_factor, f"Carbon_Emission_{f}_{u}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == "optimal":
            objective_value = model.getObjVal()
        else:
            objective_value = None

        return model.getStatus(), end_time - start_time, objective_value


if __name__ == "__main__":
    seed = 42
    parameters = {
        'n_facilities': 270,
        'n_units': 117,
        'min_transport_cost': 1620,
        'max_transport_cost': 1800,
        'min_facility_cost': 1250,
        'max_facility_cost': 5000,
        'min_facility_space': 555,
        'max_facility_space': 1350,
        'mean_demand': 1050,
        'std_dev_demand': 50,
        'graph_density': 0.38,
        'bigM': 10000,
        'min_social_impact': 50,
        'max_social_impact': 150,
        'min_carbon_emission': 10.0,
        'max_carbon_emission': 50.0,
        'social_impact_weight': 1.5,
        'carbon_emission_weight': 2.0,
        'transport_distance_factor': 0.01,
    }

    warehouse_layout_optimizer = WarehouseLayoutOptimization(parameters, seed=42)
    instance = warehouse_layout_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_layout_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")