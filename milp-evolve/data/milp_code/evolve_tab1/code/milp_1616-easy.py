import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class MobileHealthcareOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_mobile_units > 0 and self.n_neighborhoods > 0
        assert self.min_operational_cost >= 0 and self.max_operational_cost >= self.min_operational_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_supply_limit > 0 and self.max_supply_limit >= self.min_supply_limit

        operational_costs = np.random.randint(self.min_operational_cost, self.max_operational_cost + 1, self.n_mobile_units)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_mobile_units, self.n_neighborhoods))
        supply_limits = np.random.randint(self.min_supply_limit, self.max_supply_limit + 1, self.n_mobile_units)
        neighborhood_demands = np.random.randint(10, 100, self.n_neighborhoods)
        service_requirements = np.random.uniform(self.min_service_level, self.max_service_level, self.n_neighborhoods)
        distances = np.random.uniform(0, self.max_service_distance, (self.n_mobile_units, self.n_neighborhoods))
        
        neighborhood_positions = np.random.rand(self.n_neighborhoods, 2) * self.city_size
        mobile_positions = np.random.rand(self.n_mobile_units, 2) * self.city_size

        G = nx.DiGraph()
        node_pairs = []
        for m in range(self.n_mobile_units):
            for n in range(self.n_neighborhoods):
                G.add_edge(f"mobile_unit_{m}", f"neighborhood_{n}")
                node_pairs.append((f"mobile_unit_{m}", f"neighborhood_{n}"))

        return {
            "operational_costs": operational_costs,
            "transport_costs": transport_costs,
            "supply_limits": supply_limits,
            "neighborhood_demands": neighborhood_demands,
            "service_requirements": service_requirements,
            "distances": distances,
            "neighborhood_positions": neighborhood_positions,
            "mobile_positions": mobile_positions,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        transport_costs = instance['transport_costs']
        supply_limits = instance['supply_limits']
        neighborhood_demands = instance['neighborhood_demands']
        service_requirements = instance['service_requirements']
        distances = instance['distances']
        neighborhood_positions = instance['neighborhood_positions']
        mobile_positions = instance['mobile_positions']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        
        model = Model("MobileHealthcareOptimization")
        n_mobile_units = len(operational_costs)
        n_neighborhoods = len(transport_costs[0])

        # Decision variables
        open_vars = {m: model.addVar(vtype="B", name=f"MobileUnit_{m}") for m in range(n_mobile_units)}
        schedule_vars = {(u, v): model.addVar(vtype="C", name=f"Schedule_{u}_{v}") for u, v in node_pairs}
        coverage_vars = {n: model.addVar(vtype="B", name=f"Coverage_Neighborhood_{n}") for n in range(n_neighborhoods)}

        # Objective: minimize the total cost including operational costs and transport costs.
        model.setObjective(
            quicksum(operational_costs[m] * open_vars[m] for m in range(n_mobile_units)) +
            quicksum(transport_costs[m, int(v.split('_')[1])] * schedule_vars[(u, v)] for (u, v) in node_pairs for m in range(n_mobile_units) if u == f'mobile_unit_{m}'),
            "minimize"
        )

        # Supply limits for each mobile unit
        for m in range(n_mobile_units):
            model.addCons(
                quicksum(schedule_vars[(f"mobile_unit_{m}", f"neighborhood_{n}")] for n in range(n_neighborhoods)) <= supply_limits[m], 
                f"LinkedSupplyLimits_MobileUnit_{m}"
            )

        # Minimum service coverage for each neighborhood
        for n in range(n_neighborhoods):
            model.addCons(
                quicksum(schedule_vars[(u, f"neighborhood_{n}")] for u in G.predecessors(f"neighborhood_{n}")) >= service_requirements[n], 
                f"MinimumServiceCoverage_Neighborhood_{n}"
            )

        # Efficient scheduling for mobile units
        for m in range(n_mobile_units):
            for n in range(n_neighborhoods):
                model.addCons(
                    schedule_vars[(f"mobile_unit_{m}", f"neighborhood_{n}")] <= self.efficient_schedule_factor * open_vars[m], 
                    f"EfficientSchedulingParameters_MobileUnit_{m}_Neighborhood_{n}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_mobile_units': 225,
        'n_neighborhoods': 148,
        'min_transport_cost': 1500,
        'max_transport_cost': 1800,
        'min_operational_cost': 250,
        'max_operational_cost': 2250,
        'min_supply_limit': 1575,
        'max_supply_limit': 3000,
        'min_service_level': 0.8,
        'max_service_level': 315.0,
        'max_service_distance': 1250,
        'efficient_schedule_factor': 42.0,
        'city_size': 3000,
    }

    healthcare_optimizer = MobileHealthcareOptimization(parameters, seed=seed)
    instance = healthcare_optimizer.generate_instance()
    solve_status, solve_time, objective_value = healthcare_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")