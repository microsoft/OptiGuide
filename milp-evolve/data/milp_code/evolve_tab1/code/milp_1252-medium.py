import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SustainableFishing:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_crew_units > 0 and self.n_fishing_activities >= self.n_crew_units
        assert self.min_fuel_cost >= 0 and self.max_fuel_cost >= self.min_fuel_cost
        assert self.min_mkt_price >= 0 and self.max_mkt_price >= self.min_mkt_price
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        fuel_costs = np.random.randint(self.min_fuel_cost, self.max_fuel_cost + 1, self.n_crew_units)
        mkt_prices = np.random.normal(self.mean_mkt_price, self.stddev_mkt_price, (self.n_crew_units, self.n_fishing_activities)).astype(int)
        mkt_prices = np.clip(mkt_prices, self.min_mkt_price, self.max_mkt_price)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_crew_units)

        demand = {t: np.random.uniform(self.min_demand, self.max_demand) for t in range(self.n_fishing_activities)}

        G = nx.erdos_renyi_graph(n=self.n_fishing_activities, p=self.route_prob, directed=True, seed=self.seed)
        for node in G.nodes:
            G.nodes[node]['processing_rate'] = np.random.uniform(0.8, 1.2)
        for u, v in G.edges:
            G[u][v]['transition_time'] = np.random.randint(5, 15)

        fish_sustainability = np.random.uniform(self.min_sustain_limit, self.max_sustain_limit, self.n_crew_units)
        boat_activity_rate = np.random.uniform(self.min_boat_activity_rate, self.max_boat_activity_rate, self.n_crew_units)
        
        fish_limits = np.random.uniform(self.min_fish_limit, self.max_fish_limit, self.n_crew_units)
        env_impact_limits = np.random.uniform(self.min_env_impact_limit, self.max_env_impact_limit, self.n_crew_units)
        
        penalties = np.random.randint(self.min_penalty, self.max_penalty, self.n_fishing_activities)

        return {
            "fuel_costs": fuel_costs,
            "mkt_prices": mkt_prices,
            "capacities": capacities,
            "demand": demand,
            "G": G,
            "fish_sustainability": fish_sustainability,
            "boat_activity_rate": boat_activity_rate,
            "fish_limits": fish_limits,
            "env_impact_limits": env_impact_limits,
            "penalties": penalties,
        }

    def solve(self, instance):
        fuel_costs = instance['fuel_costs']
        mkt_prices = instance['mkt_prices']
        capacities = instance['capacities']
        demand = instance['demand']
        G = instance['G']
        fish_sustainability = instance['fish_sustainability']
        boat_activity_rate = instance['boat_activity_rate']
        fish_limits = instance['fish_limits']
        env_impact_limits = instance['env_impact_limits']
        penalties = instance['penalties']
        
        model = Model("SustainableFishing")
        n_crew_units = len(fuel_costs)
        n_fishing_activities = len(mkt_prices[0])

        # Decision variables
        crew_vars = {w: model.addVar(vtype="B", name=f"Crew_{w}") for w in range(n_crew_units)}
        mapping_vars = {(w, t): model.addVar(vtype="B", name=f"Crew_{w}_Activity_{t}") for w in range(n_crew_units) for t in range(n_fishing_activities)}
        dynamic_capacity_vars = {w: model.addVar(vtype="C", name=f"DynamicCapacity_{w}") for w in range(n_crew_units)}
        zero_tolerance_vars = {t: model.addVar(vtype="C", name=f"ZeroTolerance_{t}") for t in range(n_fishing_activities)}

        # New variables for quota violation costs
        quota_violation_vars = {t: model.addVar(vtype="C", name=f"QuotaViolation_{t}") for t in range(n_fishing_activities)}

        # Objective: Maximize fish revenue while minimizing fuel, quota violation, and maintenance costs
        model.setObjective(
            quicksum(demand[t] * quicksum(mapping_vars[w, t] for w in range(n_crew_units)) for t in range(n_fishing_activities)) -
            quicksum(fuel_costs[w] * crew_vars[w] for w in range(n_crew_units)) -
            quicksum(mkt_prices[w][t] * mapping_vars[w, t] for w in range(n_crew_units) for t in range(n_fishing_activities)) -
            quicksum(boat_activity_rate[w] * mapping_vars[w, t] for w in range(n_crew_units) for t in range(n_fishing_activities)) -
            quicksum(quota_violation_vars[t] * penalties[t] for t in range(n_fishing_activities)),
            "maximize"
        )

        # Constraints: Each fishing activity is assigned to exactly one crew unit
        for t in range(n_fishing_activities):
            model.addCons(quicksum(mapping_vars[w, t] for w in range(n_crew_units)) == 1, f"Activity_{t}_Assignment")

        # Constraints: Only active crew units can undertake fishing activities
        for w in range(n_crew_units):
            for t in range(n_fishing_activities):
                model.addCons(mapping_vars[w, t] <= crew_vars[w], f"Crew_{w}_Activity_{t}_Service")

        # Constraints: Units cannot exceed their dynamic capacity
        for w in range(n_crew_units):
            model.addCons(quicksum(mapping_vars[w, t] for t in range(n_fishing_activities)) <= dynamic_capacity_vars[w], f"Crew_{w}_DynamicCapacity")

        # Dynamic Capacity Constraints based on sustainability and boat activity rate
        for w in range(n_crew_units):
            model.addCons(dynamic_capacity_vars[w] == (capacities[w] - boat_activity_rate[w]), f"DynamicCapacity_{w}")

        # Fish Sustainability Constraints
        for w in range(n_crew_units):
            model.addCons(crew_vars[w] * boat_activity_rate[w] <= self.max_boat_activity_time, f"BoatActivity_{w}")

        # Fish Stock Control Constraints
        for w in range(n_crew_units):
            model.addCons(fish_sustainability[w] * quicksum(mapping_vars[w, t] for t in range(n_fishing_activities)) <= self.min_fish_impact, f"FishControl_{w}")

        # Environmental Constraints
        for w in range(n_crew_units):
            model.addCons(
                quicksum(mkt_prices[w][t] * mapping_vars[w, t] for t in range(n_fishing_activities)) <= fish_limits[w], f"FishLimit_{w}"
            )
            model.addCons(
                quicksum(mkt_prices[w][t] / 1000 * mapping_vars[w, t] for t in range(n_fishing_activities)) <= env_impact_limits[w], f"EnvImpactLimit_{w}"
            )
        
        # Quota violation constraints for activity coverage
        for t in range(n_fishing_activities):
            model.addCons(quicksum(mapping_vars[w, t] for w in range(n_crew_units)) + quota_violation_vars[t] >= 1, f"QuotaViolation_Activity_{t}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_crew_units': 18,
        'n_fishing_activities': 187,
        'min_fuel_cost': 112,
        'max_fuel_cost': 225,
        'mean_mkt_price': 2,
        'stddev_mkt_price': 60,
        'min_mkt_price': 10,
        'max_mkt_price': 225,
        'min_capacity': 56,
        'max_capacity': 1000,
        'route_prob': 0.1,
        'min_demand': 250,
        'max_demand': 843,
        'max_boat_activity_time': 450,
        'min_sustain_limit': 0.66,
        'max_sustain_limit': 2.0,
        'min_boat_activity_rate': 0.17,
        'max_boat_activity_rate': 0.73,
        'min_fish_limit': 937,
        'max_fish_limit': 4000,
        'min_env_impact_limit': 150,
        'max_env_impact_limit': 2800,
        'min_fish_impact': 600,
        'min_penalty': 0,
        'max_penalty': 150,
    }

    optimizer = SustainableFishing(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")