import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HumanitarianAidDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_regions > 0 and self.n_aid_types > 0
        assert self.min_prep_cost >= 0 and self.max_prep_cost >= self.min_prep_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost

        preparation_costs = np.random.randint(self.min_prep_cost, self.max_prep_cost + 1, self.n_aid_types)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_aid_types, self.n_regions))
        urgency_levels = np.random.randint(1, 10, self.n_regions)

        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_regions).astype(int)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_regions)

        G = nx.barabasi_albert_graph(self.n_regions, 2, seed=self.seed)
        delivery_times = {(u, v): np.random.randint(10, 60) for u, v in G.edges}
        delivery_costs = {(u, v): np.random.uniform(10.0, 60.0) for u, v in G.edges}

        weather_conditions = np.random.uniform(1.0, 3.0, self.n_regions)  # Weather impact on delivery time and cost
        security_risks = np.random.uniform(0.8, 2.0, self.n_regions)  # Security risk multipliers for delivery

        vulnerable_population_ratios = np.random.uniform(0.1, 0.5, self.n_regions)
        high_risk_zones = {u: np.random.choice([0, 1], p=[0.8, 0.2]) for u in range(self.n_regions)}  # High-risk zone indicator

        multi_modal_transportation = {u: np.random.choice(['truck', 'drone', 'boat'], p=[0.6, 0.3, 0.1]) for u in range(self.n_regions)}

        return {
            "preparation_costs": preparation_costs,
            "transport_costs": transport_costs,
            "urgency_levels": urgency_levels,
            "demands": demands,
            "capacities": capacities,
            "graph": G,
            "delivery_times": delivery_times,
            "delivery_costs": delivery_costs,
            "weather_conditions": weather_conditions,
            "security_risks": security_risks,
            "vulnerable_population_ratios": vulnerable_population_ratios,
            "high_risk_zones": high_risk_zones,
            "multi_modal_transportation": multi_modal_transportation,
        }

    # MILP modeling
    def solve(self, instance):
        preparation_costs = instance["preparation_costs"]
        transport_costs = instance["transport_costs"]
        urgency_levels = instance["urgency_levels"]
        demands = instance["demands"]
        capacities = instance["capacities"]
        G = instance["graph"]
        delivery_times = instance["delivery_times"]
        delivery_costs = instance["delivery_costs"]
        weather_conditions = instance["weather_conditions"]
        security_risks = instance["security_risks"]
        vulnerable_population_ratios = instance["vulnerable_population_ratios"]
        high_risk_zones = instance["high_risk_zones"]
        multi_modal_transportation = instance["multi_modal_transportation"]

        model = Model("HumanitarianAidDistribution")
        n_regions = len(capacities)
        n_aid_types = len(preparation_costs)

        BigM = self.bigM

        # Decision variables
        aid_vars = {a: model.addVar(vtype="B", name=f"Aid_{a}") for a in range(n_aid_types)}
        distribution_vars = {(a, r): model.addVar(vtype="B", name=f"Aid_{a}_Region_{r}") for a in range(n_aid_types) for r in range(n_regions)}
        route_vars = {(u, v): model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}

        # New variables for weather and security impact
        risk_adjusted_delivery_time = {(u, v): model.addVar(vtype="C", name=f"RiskAdjustedTime_{u}_{v}") for u, v in G.edges}
        weather_impact_vars = {(r): model.addVar(vtype="C", name=f"WeatherImpact_{r}") for r in range(n_regions)}

        # Objective function
        model.setObjective(
            quicksum(preparation_costs[a] * aid_vars[a] for a in range(n_aid_types)) +
            quicksum(transport_costs[a, r] * distribution_vars[a, r] for a in range(n_aid_types) for r in range(n_regions)) +
            quicksum(delivery_costs[(u, v)] * route_vars[(u, v)] * (weather_conditions[r] + security_risks[r]) for u, v in G.edges for r in range(n_regions) if not high_risk_zones[r]) +
            quicksum(self.priority_penalty * urgency_levels[r] * vulnerable_population_ratios[r] for r in range(n_regions)),
            "minimize"
        )

        # Constraints: Each region's demand should be met by at least one aid type
        for r in range(n_regions):
            model.addCons(quicksum(distribution_vars[a, r] for a in range(n_aid_types)) >= 1, f"Region_{r}_Demand")

        # Constraints: Regions cannot exceed their capacity
        for r in range(n_regions):
            model.addCons(quicksum(demands[r] * distribution_vars[a, r] for a in range(n_aid_types)) <= capacities[r], f"Region_{r}_Capacity")

        # Constraints: Only prepared aid types can be distributed
        for a in range(n_aid_types):
            for r in range(n_regions):
                model.addCons(distribution_vars[a, r] <= aid_vars[a], f"Aid_{a}_Deliver_{r}")

        # Constraints: Only available routes can be used
        for u, v in G.edges:
            model.addCons(route_vars[(u, v)] <= quicksum(distribution_vars[a, u] + distribution_vars[a, v] for a in range(n_aid_types)), f"Route_{u}_{v}_distribution")

        # Constraints: High-risk zones have adjusted delivery times
        for u, v in G.edges:
            model.addCons(risk_adjusted_delivery_time[(u, v)] >= delivery_times[(u, v)] * security_risks[u], f"Risk_Adjusted_Time_{u}_{v}")

        # Constraints: Weather impact on delivery
        for r in range(n_regions):
            model.addCons(weather_impact_vars[r] == weather_conditions[r], f"Weather_Impact_{r}")

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
        'n_regions': 50,
        'n_aid_types': 175,
        'min_prep_cost': 3000,
        'max_prep_cost': 5000,
        'min_transport_cost': 30,
        'max_transport_cost': 3000,
        'mean_demand': 50,
        'std_dev_demand': 60,
        'min_capacity': 900,
        'max_capacity': 3000,
        'bigM': 2000,
        'priority_penalty': 1200.0,
    }

    aid_optimizer = HumanitarianAidDistribution(parameters, seed)
    instance = aid_optimizer.generate_instance()
    solve_status, solve_time, objective_value = aid_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")