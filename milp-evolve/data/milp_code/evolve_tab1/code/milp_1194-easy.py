import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class BeekeepingPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_beekeeping_units > 0 and self.n_tasks >= self.n_beekeeping_units
        assert self.min_nutrient_cost >= 0 and self.max_nutrient_cost >= self.min_nutrient_cost
        assert self.min_honey_cost >= 0 and self.max_honey_cost >= self.min_honey_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        nutrient_costs = np.random.randint(self.min_nutrient_cost, self.max_nutrient_cost + 1, self.n_beekeeping_units)
        honey_costs = np.random.normal(self.mean_honey_cost, self.stddev_honey_cost, (self.n_beekeeping_units, self.n_tasks)).astype(int)
        honey_costs = np.clip(honey_costs, self.min_honey_cost, self.max_honey_cost)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_beekeeping_units)

        market_demand = {t: np.random.uniform(self.min_demand, self.max_demand) for t in range(self.n_tasks)}

        G = nx.erdos_renyi_graph(n=self.n_tasks, p=self.route_prob, directed=True, seed=self.seed)
        for node in G.nodes:
            G.nodes[node]['processing_rate'] = np.random.uniform(0.8, 1.2)
        for u, v in G.edges:
            G[u][v]['transition_time'] = np.random.randint(5, 15)

        health_metrics = np.random.uniform(self.min_health_metric, self.max_health_metric, self.n_beekeeping_units)
        bear_activity_rate = np.random.uniform(self.min_bear_activity_rate, self.max_bear_activity_rate, self.n_beekeeping_units)
        
        honey_limits = np.random.uniform(self.min_honey_limit, self.max_honey_limit, self.n_beekeeping_units)
        health_impact_limits = np.random.uniform(self.min_health_impact_limit, self.max_health_impact_limit, self.n_beekeeping_units)

        return {
            "nutrient_costs": nutrient_costs,
            "honey_costs": honey_costs,
            "capacities": capacities,
            "market_demand": market_demand,
            "G": G,
            "health_metrics": health_metrics,
            "bear_activity_rate": bear_activity_rate,
            "honey_limits": honey_limits,
            "health_impact_limits": health_impact_limits,
        }

    def solve(self, instance):
        nutrient_costs = instance['nutrient_costs']
        honey_costs = instance['honey_costs']
        capacities = instance['capacities']
        market_demand = instance['market_demand']
        G = instance['G']
        health_metrics = instance['health_metrics']
        bear_activity_rate = instance['bear_activity_rate']
        honey_limits = instance['honey_limits']
        health_impact_limits = instance['health_impact_limits']

        model = Model("BeekeepingPlanning")
        n_beekeeping_units = len(nutrient_costs)
        n_tasks = len(honey_costs[0])

        # Decision variables
        unit_vars = {w: model.addVar(vtype="B", name=f"BeekeepingUnit_{w}") for w in range(n_beekeeping_units)}
        allocation_vars = {(w, t): model.addVar(vtype="B", name=f"BeekeepingUnit_{w}_Task_{t}") for w in range(n_beekeeping_units) for t in range(n_tasks)}
        dynamic_capacity_vars = {w: model.addVar(vtype="C", name=f"DynamicCapacity_{w}") for w in range(n_beekeeping_units)}
        health_shift_vars = {t: model.addVar(vtype="C", name=f"HealthShift_{t}") for t in range(n_tasks)}

        # Objective: Maximize honey production while minimizing nutrient, honey processing costs and health impacts
        model.setObjective(
            quicksum(market_demand[t] * quicksum(allocation_vars[w, t] for w in range(n_beekeeping_units)) for t in range(n_tasks)) -
            quicksum(nutrient_costs[w] * unit_vars[w] for w in range(n_beekeeping_units)) -
            quicksum(honey_costs[w][t] * allocation_vars[w, t] for w in range(n_beekeeping_units) for t in range(n_tasks)) -
            quicksum(bear_activity_rate[w] * allocation_vars[w, t] for w in range(n_beekeeping_units) for t in range(n_tasks)),
            "maximize"
        )

        # Constraints: Each task is assigned to exactly one beekeeping unit
        for t in range(n_tasks):
            model.addCons(quicksum(allocation_vars[w, t] for w in range(n_beekeeping_units)) == 1, f"Task_{t}_Assignment")

        # Constraints: Only active units can process honey
        for w in range(n_beekeeping_units):
            for t in range(n_tasks):
                model.addCons(allocation_vars[w, t] <= unit_vars[w], f"BeekeepingUnit_{w}_Task_{t}_Service")

        # Constraints: Units cannot exceed their dynamic honey production capacity
        for w in range(n_beekeeping_units):
            model.addCons(quicksum(allocation_vars[w, t] for t in range(n_tasks)) <= dynamic_capacity_vars[w], f"BeekeepingUnit_{w}_DynamicCapacity")

        # Dynamic Capacity Constraints based on health metrics and bear activity rate
        for w in range(n_beekeeping_units):
            model.addCons(dynamic_capacity_vars[w] == (capacities[w] - bear_activity_rate[w]), f"DynamicCapacity_{w}")

        # Health Maintenance Constraints
        for w in range(n_beekeeping_units):
            model.addCons(unit_vars[w] * bear_activity_rate[w] <= self.max_bear_activity_time, f"BearActivity_{w}")

        # Health Control Constraints
        for w in range(n_beekeeping_units):
            model.addCons(health_metrics[w] * quicksum(allocation_vars[w, t] for t in range(n_tasks)) <= self.min_health_impact, f"HealthControl_{w}")

        # Environmental Constraints
        for w in range(n_beekeeping_units):
            model.addCons(
                quicksum(honey_costs[w][t] * allocation_vars[w, t] for t in range(n_tasks)) <= honey_limits[w], f"HoneyLimit_{w}"
            )
            model.addCons(
                quicksum(honey_costs[w][t] / 1000 * allocation_vars[w, t] for t in range(n_tasks)) <= health_impact_limits[w], f"HealthImpactLimit_{w}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_beekeeping_units': 18,
        'n_tasks': 375,
        'min_nutrient_cost': 150,
        'max_nutrient_cost': 225,
        'mean_honey_cost': 1,
        'stddev_honey_cost': 12,
        'min_honey_cost': 5,
        'max_honey_cost': 15,
        'min_capacity': 75,
        'max_capacity': 1000,
        'route_prob': 0.66,
        'min_demand': 500,
        'max_demand': 1125,
        'max_bear_activity_time': 600,
        'min_health_metric': 0.8,
        'max_health_metric': 1.0,
        'min_bear_activity_rate': 0.17,
        'max_bear_activity_rate': 0.8,
        'min_honey_limit': 1875,
        'max_honey_limit': 4000,
        'min_health_impact_limit': 50,
        'max_health_impact_limit': 400,
        'min_health_impact': 600,
    }

    optimizer = BeekeepingPlanning(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")