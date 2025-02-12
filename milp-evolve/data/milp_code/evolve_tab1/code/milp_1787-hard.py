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
        mutual_exclusivity_pairs = [(random.randint(0, self.n_generators - 1), random.randint(0, self.n_generators - 1)) for _ in range(self.n_exclusive_pairs)]

        maintenance_costs = np.random.uniform(100, 500, size=self.n_generators).tolist()

        piecewise_intervals = [self.gen_capacity_interval]*self.n_generators  # creating piecewise intervals for each generator
        piecewise_costs = []
        for _ in range(self.n_generators):
            costs = np.random.uniform(0.5, 2.0, self.gen_capacity_interval + 1).tolist()  # random piecewise costs
            piecewise_costs.append(costs)

        # Facility data generation
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.gamma(shape=2.0, scale=1.0, size=n_facilities).tolist()
        assignment_cost = np.random.normal(loc=5, scale=2, size=self.n_generators).tolist()
        capacity = np.random.randint(10, 50, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()
        throughput = np.random.uniform(1.0, 5.0, size=self.n_generators).tolist()

        return {
            "gen_costs": gen_costs,
            "trans_costs": trans_costs,
            "gen_capacities": gen_capacities,
            "load_demands": load_demands,
            "shed_penalties": shed_penalties,
            "emission_penalties": emission_penalties,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "maintenance_costs": maintenance_costs,
            "piecewise_intervals": piecewise_intervals,
            "piecewise_costs": piecewise_costs,
            "n_facilities": n_facilities,
            "operating_cost": operating_cost,
            "assignment_cost": assignment_cost,
            "capacity": capacity,
            "setup_cost": setup_cost,
            "throughput": throughput,
        }

    def solve(self, instance):
        gen_costs = instance['gen_costs']
        trans_costs = instance['trans_costs']
        gen_capacities = instance['gen_capacities']
        load_demands = instance['load_demands']
        shed_penalties = instance['shed_penalties']
        emission_penalties = instance['emission_penalties']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        maintenance_costs = instance['maintenance_costs']
        piecewise_intervals = instance['piecewise_intervals']
        piecewise_costs = instance['piecewise_costs']
        n_facilities = instance["n_facilities"]
        operating_cost = instance["operating_cost"]
        assignment_cost = instance["assignment_cost"]
        capacity = instance["capacity"]
        setup_cost = instance["setup_cost"]
        throughput = instance["throughput"]

        model = Model("ElectricalLoadBalancingOptimization")
        n_generators = len(gen_costs)
        n_loads = len(trans_costs[0])

        # Decision variables
        gen_vars = {g: model.addVar(vtype="B", name=f"Generator_{g}") for g in range(n_generators)}
        load_vars = {(g, l): model.addVar(vtype="C", name=f"Load_{g}_{l}") for g in range(n_generators) for l in range(n_loads)}
        shed_vars = {l: model.addVar(vtype="C", name=f"Shed_{l}") for l in range(n_loads)}

        # Additional variables for emissions
        emission_vars = {g: model.addVar(vtype="C", name=f"Emission_{g}") for g in range(n_generators)}

        # Mutual Exclusivity Variables using logical conditions
        mutual_exclusivity_vars = {(g1, g2): model.addVar(vtype="B", name=f"MutualEx_{g1}_{g2}") for (g1, g2) in mutual_exclusivity_pairs}

        # New variables for piecewise linear costs
        piecewise_vars = {(g, k): model.addVar(vtype="B", name=f"Piecewise_{g}_{k}") for g in range(n_generators) for k in range(piecewise_intervals[g])}
        
        # Piecewise cost variables
        piecewise_cost_weight_vars = {(g, k): model.addVar(vtype="C", name=f"PiecewiseWeight_{g}_{k}") for g in range(n_generators) for k in range(piecewise_intervals[g])}

        # Facility related variables
        facility_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        assign_vars = {(g, j): model.addVar(vtype="B", name=f"Assign_{g}_{j}") for g in range(n_generators) for j in range(n_facilities)}

        # Throughput assignment variables
        throughput_vars = {g: model.addVar(vtype="C", name=f"Throughput_{g}") for g in range(n_generators)}

        # Objective: minimize the total cost
        model.setObjective(
            quicksum(piecewise_cost_weight_vars[(g, k)] * piecewise_costs[g][k] for g in range(n_generators) for k in range(piecewise_intervals[g])) +
            quicksum(trans_costs[g, l] * load_vars[(g, l)] for g in range(n_generators) for l in range(n_loads)) +
            quicksum(shed_penalties[l] * shed_vars[l] for l in range(n_loads)) +
            quicksum(emission_penalties[g] * emission_vars[g] for g in range(n_generators)) +
            quicksum(maintenance_costs[g] * gen_vars[g] for g in range(n_generators)) +
            quicksum(operating_cost[j] * facility_vars[j] for j in range(n_facilities)) +
            quicksum(assignment_cost[g] * assign_vars[(g, j)] for g in range(n_generators) for j in range(n_facilities)) +
            quicksum(setup_cost[j] * facility_vars[j] for j in range(n_facilities)),
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

        # Mutual Exclusivity Constraints using Logical Conditions
        for g1, g2 in mutual_exclusivity_pairs:
            model.addCons(mutual_exclusivity_vars[(g1, g2)] == gen_vars[g1] + gen_vars[g2], f"MutualExclusivity_{g1}_{g2}")
            model.addCons(mutual_exclusivity_vars[(g1, g2)] <= 1, f"MutualExclusivityLimit_{g1}_{g2}")

        # Convex Hull Formulation for piecewise linear costs
        for g in range(n_generators):
            intervals = piecewise_intervals[g]
            for k in range(intervals):
                model.addCons(piecewise_vars[(g, k)] <= gen_vars[g], f"PiecewiseGenLimit_{g}_{k}")

            model.addCons(
                quicksum(piecewise_vars[(g, k)] for k in range(intervals)) == gen_vars[g], 
                f"ConvexHull_{g}"
            )

            model.addCons(
                quicksum(piecewise_cost_weight_vars[(g, k)] for k in range(intervals)) == 
                quicksum(load_vars[(g, l)] for l in range(n_loads)), 
                f"PiecewiseWeightConstr_{g}"
            )

        # Constraints: Facilities cannot exceed their capacities
        for j in range(n_facilities):
            model.addCons(quicksum(assign_vars[(g, j)] for g in range(n_generators)) <= capacity[j] * facility_vars[j], f"FacilityCapacity_{j}")

        # Assignment Constraints for Throughput
        max_throughput = np.max(throughput) * n_generators
        for j in range(n_facilities):
            model.addCons(quicksum(throughput[g] * assign_vars[(g, j)] for g in range(n_generators)) <= max_throughput * facility_vars[j], f"MaxThroughput_{j}")

        # Constraints: Each generator should be assigned to at most one facility
        for g in range(n_generators):
            model.addCons(quicksum(assign_vars[(g, j)] for j in range(n_facilities)) == gen_vars[g], f"Assign_{g}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_generators': 160,
        'n_loads': 350,
        'min_trans_cost': 100,
        'max_trans_cost': 525,
        'min_gen_cost': 630,
        'max_gen_cost': 840,
        'min_gen_capacity': 200,
        'max_gen_capacity': 700,
        'min_shed_penalty': 157,
        'max_shed_penalty': 450,
        'min_emission_penalty': 45,
        'max_emission_penalty': 150,
        'n_exclusive_pairs': 500,
        'gen_capacity_interval': 3,
        'facility_min_count': 3,
        'facility_max_count': 75,
    }

    optimizer = ElectricalLoadBalancingOptimization(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")