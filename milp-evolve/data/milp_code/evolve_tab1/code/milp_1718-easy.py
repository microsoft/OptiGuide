import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyResponseOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_er_units > 0 and self.n_zones > 0
        assert self.min_operational_cost >= 0 and self.max_operational_cost >= self.min_operational_cost
        assert self.min_travel_cost >= 0 and self.max_travel_cost >= self.min_travel_cost
        assert self.min_operational_hours > 0 and self.max_operational_hours >= self.min_operational_hours

        operational_costs = np.random.randint(self.min_operational_cost, self.max_operational_cost + 1, self.n_er_units)
        travel_costs = np.random.randint(self.min_travel_cost, self.max_travel_cost + 1, (self.n_er_units, self.n_zones))
        operational_hours = np.random.randint(self.min_operational_hours, self.max_operational_hours + 1, self.n_er_units)
        zone_demands = np.random.randint(1, 10, self.n_zones)
        evacuation_safety = np.random.uniform(self.min_safety_limit, self.max_safety_limit, self.n_er_units)
        zone_awareness_level = np.random.uniform(0, 1, (self.n_er_units, self.n_zones))

        G = nx.Graph()
        er_zone_pairs = []
        for p in range(self.n_er_units):
            for d in range(self.n_zones):
                G.add_edge(f"unit_{p}", f"zone_{d}")
                er_zone_pairs.append((f"unit_{p}", f"zone_{d}"))

        return {
            "operational_costs": operational_costs,
            "travel_costs": travel_costs,
            "operational_hours": operational_hours,
            "zone_demands": zone_demands,
            "evacuation_safety": evacuation_safety,
            "zone_awareness_level": zone_awareness_level,
            "graph": G,
            "er_zone_pairs": er_zone_pairs
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        travel_costs = instance['travel_costs']
        operational_hours = instance['operational_hours']
        zone_demands = instance['zone_demands']
        evacuation_safety = instance['evacuation_safety']
        zone_awareness_level = instance['zone_awareness_level']
        G = instance['graph']
        er_zone_pairs = instance['er_zone_pairs']

        model = Model("EmergencyResponseOptimization")
        n_er_units = len(operational_costs)
        n_zones = len(travel_costs[0])

        # Decision variables
        er_unit_vars = {p: model.addVar(vtype="B", name=f"AllocateUnit_{p}") for p in range(n_er_units)}
        route_vars = {(u, v): model.addVar(vtype="C", name=f"Route_{u}_{v}") for u, v in er_zone_pairs}
        operational_vars = {p: model.addVar(vtype="C", name=f"OperationalHours_{p}") for p in range(n_er_units)}

        # Objective: minimize the total cost including operational costs and travel costs.
        model.setObjective(
            quicksum(operational_costs[p] * er_unit_vars[p] for p in range(n_er_units)) +
            quicksum(travel_costs[p, int(v.split('_')[1])] * route_vars[(u, v)] for (u, v) in er_zone_pairs for p in range(n_er_units) if u == f'unit_{p}'),
            "minimize"
        )

        # Route constraint for each zone
        for d in range(n_zones):
            model.addCons(
                quicksum(route_vars[(u, f"zone_{d}")] for u in G.neighbors(f"zone_{d}")) == zone_demands[d], 
                f"Zone_{d}_RouteRequirements"
            )

        # Constraints: Zones receive ERU service according to the safety limit
        for p in range(n_er_units):
            for d in range(n_zones):
                model.addCons(
                    route_vars[(f"unit_{p}", f"zone_{d}")] <= evacuation_safety[p] * er_unit_vars[p], 
                    f"Unit_{p}_RouteLimitBySafety_{d}"
                )

        # Constraints: ERUs cannot exceed their operational hours
        for p in range(n_er_units):
            model.addCons(
                quicksum(route_vars[(f"unit_{p}", f"zone_{d}")] for d in range(n_zones)) <= operational_hours[p], 
                f"Unit_{p}_MaxOperationalHours"
            )

        # Zone awareness level constraints
        for p in range(n_er_units):
            for d in range(n_zones):
                model.addCons(
                    route_vars[(f"unit_{p}", f"zone_{d}")] <= (1 - zone_awareness_level[p, d]) * er_unit_vars[p],
                    f"Unit_{p}_ZoneAwareness_{d}"
                )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_er_units': 2700,
        'n_zones': 7,
        'min_travel_cost': 562,
        'max_travel_cost': 3000,
        'min_operational_cost': 2250,
        'max_operational_cost': 3000,
        'min_operational_hours': 375,
        'max_operational_hours': 450,
        'min_safety_limit': 10,
        'max_safety_limit': 45,
    }

    optimizer = EmergencyResponseOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")