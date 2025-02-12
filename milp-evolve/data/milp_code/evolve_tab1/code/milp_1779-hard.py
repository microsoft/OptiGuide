import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ElectricalLoadBalancingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_generators > 0 and self.n_loads > 0
        assert self.min_gen_cost >= 0 and self.max_gen_cost >= self.min_gen_cost
        assert self.min_trans_cost >= 0 and self.max_trans_cost >= self.min_trans_cost
        assert self.min_gen_capacity > 0 and self.max_gen_capacity >= self.min_gen_capacity
        assert self.min_emission_penalty >= 0 and self.max_emission_penalty >= self.min_emission_penalty

        # Generator & Transmission costs
        gen_costs = np.random.randint(self.min_gen_cost, self.max_gen_cost + 1, self.n_generators)
        trans_costs = np.random.randint(self.min_trans_cost, self.max_trans_cost + 1, (self.n_generators, self.n_loads))
        
        # Dynamic capacities and demands
        seasonality_factors = np.random.uniform(0.5, 1.5, self.n_loads)  # seasonal factors for load scaling
        weather_impact = np.random.uniform(0.8, 1.2, self.n_generators)  # weather impact for generation capacity
        base_gen_capacities = np.random.randint(self.min_gen_capacity, self.max_gen_capacity + 1, self.n_generators)
        gen_capacities = (base_gen_capacities * weather_impact).astype(int)
        load_demands = (np.random.randint(1, 10, self.n_loads) * seasonality_factors).astype(int)

        # Penalties and costs
        shed_penalties = np.random.uniform(self.min_shed_penalty, self.max_shed_penalty, self.n_loads)
        emission_penalties = np.random.uniform(self.min_emission_penalty, self.max_emission_penalty, self.n_generators)
        mutual_dependency_pairs = [(random.randint(0, self.n_generators - 1), random.randint(0, self.n_generators - 1)) for _ in range(self.n_dependency_pairs)]
        
        # Maintenance costs with dynamic component
        maintenance_costs = (np.random.uniform(100, 500, size=self.n_generators) * weather_impact).tolist()

        # Multi-stage costs
        multi_stage_intervals = [self.gen_capacity_interval]*self.n_generators
        multi_stage_costs = []
        for _ in range(self.n_generators):
            costs = np.random.uniform(1.0, 3.0, self.gen_capacity_interval + 1).tolist()
            multi_stage_costs.append(costs)

        return {
            "gen_costs": gen_costs,
            "trans_costs": trans_costs,
            "gen_capacities": gen_capacities,
            "load_demands": load_demands,
            "shed_penalties": shed_penalties,
            "emission_penalties": emission_penalties,
            "mutual_dependency_pairs": mutual_dependency_pairs,
            "maintenance_costs": maintenance_costs,
            "multi_stage_intervals": multi_stage_intervals,
            "multi_stage_costs": multi_stage_costs,
        }

    def solve(self, instance):
        gen_costs = instance['gen_costs']
        trans_costs = instance['trans_costs']
        gen_capacities = instance['gen_capacities']
        load_demands = instance['load_demands']
        shed_penalties = instance['shed_penalties']
        emission_penalties = instance['emission_penalties']
        mutual_dependency_pairs = instance['mutual_dependency_pairs']
        maintenance_costs = instance['maintenance_costs']
        multi_stage_intervals = instance['multi_stage_intervals']
        multi_stage_costs = instance['multi_stage_costs']

        model = Model("ElectricalLoadBalancingOptimization")
        n_generators = len(gen_costs)
        n_loads = len(trans_costs[0])

        # Decision variables
        gen_vars = {g: model.addVar(vtype="B", name=f"Generator_{g}") for g in range(n_generators)}
        load_vars = {(g, l): model.addVar(vtype="C", name=f"Load_{g}_{l}") for g in range(n_generators) for l in range(n_loads)}
        shed_vars = {l: model.addVar(vtype="C", name=f"Shed_{l}") for l in range(n_loads)}

        # Emission variables
        emission_vars = {g: model.addVar(vtype="C", name=f"Emission_{g}") for g in range(n_generators)}

        # Mutual dependency variables
        dependency_vars = {(g1, g2): model.addVar(vtype="B", name=f"Dependency_{g1}_{g2}") for (g1, g2) in mutual_dependency_pairs}

        # Multi-stage cost variables
        stage_vars = {(g, k): model.addVar(vtype="B", name=f"Stage_{g}_{k}") for g in range(n_generators) for k in range(multi_stage_intervals[g])}
        stage_weight_vars = {(g, k): model.addVar(vtype="C", name=f"StageWeight_{g}_{k}") for g in range(n_generators) for k in range(multi_stage_intervals[g])}

        # Objective: minimize the total cost
        model.setObjective(
            quicksum(stage_weight_vars[(g, k)] * multi_stage_costs[g][k] for g in range(n_generators) for k in range(multi_stage_intervals[g])) +
            quicksum(trans_costs[g, l] * load_vars[(g, l)] for g in range(n_generators) for l in range(n_loads)) +
            quicksum(shed_penalties[l] * shed_vars[l] for l in range(n_loads)) +
            quicksum(emission_penalties[g] * emission_vars[g] for g in range(n_generators)) +
            quicksum(maintenance_costs[g] * gen_vars[g] for g in range(n_generators)),
            "minimize"
        )

        # Load balancing constraint for each load
        for l in range(n_loads):
            model.addCons(
                quicksum(load_vars[(g, l)] for g in range(n_generators)) + shed_vars[l] == load_demands[l], 
                f"Load_{l}_Balance"
            )

        # Constraints: Generators cannot exceed their capacities
        for g in range(n_generators):
            model.addCons(
                quicksum(load_vars[(g, l)] for l in range(n_loads)) <= gen_capacities[g] * gen_vars[g], 
                f"Generator_{g}_Capacity"
            )

        # Constraints: Load shedding variables must be non-negative
        for l in range(n_loads):
            model.addCons(shed_vars[l] >= 0, f"Shed_{l}_NonNegative")

        # Emissions constraints
        for g in range(n_generators):
            model.addCons(emission_vars[g] >= quicksum(load_vars[(g, l)] for l in range(n_loads)), f"Emission_{g}_Constraint")

        # Mutual dependency constraints
        for g1, g2 in mutual_dependency_pairs:
            model.addCons(dependency_vars[(g1, g2)] == gen_vars[g1] + gen_vars[g2], f"Dependency_{g1}_{g2}")
            model.addCons(dependency_vars[(g1, g2)] <= 1, f"DependencyLimit_{g1}_{g2}")

        # Multi-stage cost constraints
        for g in range(n_generators):
            intervals = multi_stage_intervals[g]
            for k in range(intervals):
                model.addCons(stage_vars[(g, k)] <= gen_vars[g], f"StageGenLimit_{g}_{k}")
            model.addCons(
                quicksum(stage_vars[(g, k)] for k in range(intervals)) == gen_vars[g], 
                f"MultiStageHull_{g}"
            )
            model.addCons(
                quicksum(stage_weight_vars[(g, k)] for k in range(intervals)) == 
                quicksum(load_vars[(g, l)] for l in range(n_loads)), 
                f"StageWeightConstr_{g}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_generators': 160,
        'n_loads': 350,
        'min_trans_cost': 300,
        'max_trans_cost': 1575,
        'min_gen_cost': 630,
        'max_gen_cost': 2520,
        'min_gen_capacity': 400,
        'max_gen_capacity': 525,
        'min_shed_penalty': 117,
        'max_shed_penalty': 2250,
        'min_emission_penalty': 135,
        'max_emission_penalty': 450,
        'n_dependency_pairs': 1000,
        'gen_capacity_interval': 6,
    }

    optimizer = ElectricalLoadBalancingOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")