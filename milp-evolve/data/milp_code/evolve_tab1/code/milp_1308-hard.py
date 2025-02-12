import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MealServiceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_halls > 0 and self.n_meals > 0
        assert self.min_prep_cost >= 0 and self.max_prep_cost >= self.min_prep_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost

        preparation_costs = np.random.randint(self.min_prep_cost, self.max_prep_cost + 1, self.n_meals)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_meals, self.n_halls))
        dietary_needs = np.random.randint(self.min_dietary_requirement, self.max_dietary_requirement + 1, self.n_meals)

        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_halls).astype(int)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_halls)

        G = nx.barabasi_albert_graph(self.n_halls, 2, seed=self.seed)
        delivery_times = {(u, v): np.random.randint(10, 60) for u, v in G.edges}
        delivery_costs = {(u, v): np.random.uniform(10.0, 60.0) for u, v in G.edges}

        freshness_windows = np.random.randint(10, 30, self.n_halls)  # Freshness window for each hall
        traffic_conditions = {(u, v): np.random.uniform(0.5, 2.0) for u, v in G.edges}  # Traffic condition multipliers
        road_closures = {(u, v): np.random.choice([0, 1], p=[0.95, 0.05]) for u, v in G.edges}  # Probability of road closure

        refrigerated_storage = np.random.randint(50, 150, self.n_halls)  # Refrigerated storage capacity at each hall

        return {
            "preparation_costs": preparation_costs,
            "transport_costs": transport_costs,
            "dietary_needs": dietary_needs,
            "demands": demands,
            "capacities": capacities,
            "graph": G,
            "delivery_times": delivery_times,
            "delivery_costs": delivery_costs,
            "freshness_windows": freshness_windows,
            "traffic_conditions": traffic_conditions,
            "road_closures": road_closures,
            "refrigerated_storage": refrigerated_storage
        }

    # MILP modeling
    def solve(self, instance):
        preparation_costs = instance["preparation_costs"]
        transport_costs = instance["transport_costs"]
        dietary_needs = instance["dietary_needs"]
        demands = instance["demands"]
        capacities = instance["capacities"]
        G = instance["graph"]
        delivery_times = instance["delivery_times"]
        delivery_costs = instance["delivery_costs"]
        freshness_windows = instance["freshness_windows"]
        traffic_conditions = instance["traffic_conditions"]
        road_closures = instance["road_closures"]
        refrigerated_storage = instance["refrigerated_storage"]

        model = Model("MealServiceOptimization")
        n_halls = len(capacities)
        n_meals = len(preparation_costs)

        BigM = self.bigM

        # Decision variables
        meal_vars = {m: model.addVar(vtype="B", name=f"Meal_{m}") for m in range(n_meals)}
        distribution_vars = {(m, h): model.addVar(vtype="B", name=f"Meal_{m}_Hall_{h}") for m in range(n_meals) for h in range(n_halls)}
        zone_vars = {(u, v): model.addVar(vtype="B", name=f"Zone_{u}_{v}") for u, v in G.edges}
        nutrient_vars = {m: model.addVar(vtype="B", name=f"Nutrient_{m}_Fulfilled") for m in range(n_meals)}
        
        # New variables for freshness window and refrigerated storage
        freshness_vars = {(m, h): model.addVar(vtype="C", lb=0, ub=freshness_windows[h], name=f"Freshness_{m}_Hall_{h}") for m in range(n_meals) for h in range(n_halls)}
        storage_vars = {(h): model.addVar(vtype="I", lb=0, ub=refrigerated_storage[h], name=f"Storage_{h}") for h in range(n_halls)}

        # Objective function
        model.setObjective(
            quicksum(preparation_costs[m] * meal_vars[m] for m in range(n_meals)) +
            quicksum(transport_costs[m, h] * distribution_vars[m, h] for m in range(n_meals) for h in range(n_halls)) +
            quicksum(delivery_costs[(u, v)] * zone_vars[(u, v)] * traffic_conditions[(u, v)] for u, v in G.edges if not road_closures[(u, v)]) +
            quicksum(self.diet_penalty * nutrient_vars[m] for m in range(n_meals)),
            "minimize"
        )

        # Constraints: Each hall's demand should be met by at least one meal
        for h in range(n_halls):
            model.addCons(quicksum(distribution_vars[m, h] for m in range(n_meals)) >= 1, f"Hall_{h}_Demand")

        # Constraints: Halls cannot exceed their capacity
        for h in range(n_halls):
            model.addCons(quicksum(demands[h] * distribution_vars[m, h] for m in range(n_meals)) <= capacities[h], f"Hall_{h}_Capacity")

        # Constraints: Only prepared meals can be distributed
        for m in range(n_meals):
            for h in range(n_halls):
                model.addCons(distribution_vars[m, h] <= meal_vars[m], f"Meal_{m}_Serve_{h}")

        # Constraints: Only available zones can route edges
        for u, v in G.edges:
            model.addCons(zone_vars[(u, v)] <= quicksum(distribution_vars[m, u] + distribution_vars[m, v] for m in range(n_meals)), f"Zone_{u}_{v}_distribution")
        
        # Freshness window constraints
        for h in range(n_halls):
            for m in range(n_meals):
                model.addCons(freshness_vars[(m, h)] <= freshness_windows[h], f"Freshness_Window_Hall_{h}_Meal_{m}")

        # Refrigerated storage constraints
        for h in range(n_halls):
            model.addCons(quicksum(distribution_vars[m, h] for m in range(n_meals)) <= storage_vars[h], f"Storage_Capacity_Hall_{h}")

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
        'n_halls': 450,
        'n_meals': 15,
        'min_prep_cost': 1500,
        'max_prep_cost': 1800,
        'min_transport_cost': 2,
        'max_transport_cost': 400,
        'min_dietary_requirement': 45,
        'max_dietary_requirement': 62,
        'mean_demand': 10,
        'std_dev_demand': 10,
        'min_capacity': 2250,
        'max_capacity': 2500,
        'bigM': 450,
        'diet_penalty': 225.0,
        'max_storage_capacity': 150,
    }

    meal_optimizer = MealServiceOptimization(parameters, seed)
    instance = meal_optimizer.generate_instance()
    solve_status, solve_time, objective_value = meal_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")