import random
import time
import numpy as np
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

        gen_costs = np.random.randint(self.min_gen_cost, self.max_gen_cost + 1, self.n_generators)
        trans_costs = np.random.randint(self.min_trans_cost, self.max_trans_cost + 1, (self.n_generators, self.n_loads))
        gen_capacities = np.random.randint(self.min_gen_capacity, self.max_gen_capacity + 1, self.n_generators)
        load_demands = np.random.randint(1, 10, self.n_loads)
        shed_penalties = np.random.uniform(self.min_shed_penalty, self.max_shed_penalty, self.n_loads)
        emission_penalties = np.random.uniform(self.min_emission_penalty, self.max_emission_penalty, self.n_generators)
        
        return {
            "gen_costs": gen_costs,
            "trans_costs": trans_costs,
            "gen_capacities": gen_capacities,
            "load_demands": load_demands,
            "shed_penalties": shed_penalties,
            "emission_penalties": emission_penalties,
        }

    def solve(self, instance):
        gen_costs = instance['gen_costs']
        trans_costs = instance['trans_costs']
        gen_capacities = instance['gen_capacities']
        load_demands = instance['load_demands']
        shed_penalties = instance['shed_penalties']
        emission_penalties = instance['emission_penalties']

        model = Model("ElectricalLoadBalancingOptimization")
        n_generators = len(gen_costs)
        n_loads = len(trans_costs[0])

        # Decision variables
        gen_vars = {g: model.addVar(vtype="B", name=f"Generator_{g}") for g in range(n_generators)}
        load_vars = {(g, l): model.addVar(vtype="C", name=f"Load_{g}_{l}") for g in range(n_generators) for l in range(n_loads)}
        shed_vars = {l: model.addVar(vtype="C", name=f"Shed_{l}") for l in range(n_loads)}

        # Additional variables for emissions
        emission_vars = {g: model.addVar(vtype="C", name=f"Emission_{g}") for g in range(n_generators)}

        # Objective: minimize the total cost
        model.setObjective(
            quicksum(gen_costs[g] * gen_vars[g] for g in range(n_generators)) +
            quicksum(trans_costs[g, l] * load_vars[(g, l)] for g in range(n_generators) for l in range(n_loads)) +
            quicksum(shed_penalties[l] * shed_vars[l] for l in range(n_loads)) +
            quicksum(emission_penalties[g] * emission_vars[g] for g in range(n_generators)),
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

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_generators': 50,
        'n_loads': 1000,
        'min_trans_cost': 800,
        'max_trans_cost': 1575,
        'min_gen_cost': 1890,
        'max_gen_cost': 2520,
        'min_gen_capacity': 600,
        'max_gen_capacity': 700,
        'min_shed_penalty': 157,
        'max_shed_penalty': 2700,
        'min_emission_penalty': 90,
        'max_emission_penalty': 150,
    }

    optimizer = ElectricalLoadBalancingOptimization(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")