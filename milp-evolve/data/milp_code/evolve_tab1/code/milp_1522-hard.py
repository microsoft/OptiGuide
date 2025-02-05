import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class MultipleKnapsack:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        weights = np.random.normal(loc=self.weight_mean, scale=self.weight_std, size=self.number_of_items).astype(int)
        profits = weights + np.random.normal(loc=self.profit_mean_shift, scale=self.profit_std, size=self.number_of_items).astype(int)

        # Ensure non-negative values
        weights = np.clip(weights, self.min_range, self.max_range)
        profits = np.clip(profits, self.min_range, self.max_range)

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        res = {'weights': weights, 'profits': profits, 'capacities': capacities}

        # Generating random cliques
        cliques = []
        for _ in range(self.num_cliques):
            clique_size = np.random.randint(2, self.max_clique_size + 1)
            clique = np.random.choice(self.number_of_items, size=clique_size, replace=False)
            cliques.append(clique.tolist())

        res['cliques'] = cliques

        ### New data generation from Waste Management Optimization ###
        # Generate zones and emissions
        self.num_zones = np.random.randint(self.min_zones, self.max_zones)
        emissions = np.random.uniform(self.min_emission, self.max_emission, size=self.number_of_items)
        zone_capacities = np.random.uniform(self.min_zone_capacity, self.max_zone_capacity, size=self.num_zones).astype(int)
        fuel_consumption = np.random.uniform(self.min_fuel, self.max_fuel, size=self.number_of_items)

        res['emissions'] = emissions
        res['zone_capacities'] = zone_capacities
        res['fuel_consumption'] = fuel_consumption
        res['num_zones'] = self.num_zones
        res['sustainability_budget'] = self.sustainability_budget
        
        return res        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        cliques = instance['cliques']
        emissions = instance['emissions']
        zone_capacities = instance['zone_capacities']
        fuel_consumption = instance['fuel_consumption']
        num_zones = instance['num_zones']
        sustainability_budget = instance['sustainability_budget']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsack")
        var_names = {}
        z = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
            z[i] = model.addVar(vtype="B", name=f"z_{i}")
        
        # New zone assignment variables: y[i][j] = 1 if item i is placed in zone j
        y = {}
        for i in range(number_of_items):
            for j in range(num_zones):
                y[(i, j)] = model.addVar(vtype="B", name=f"y_{i}_{j}")

        # Emission variable
        emission_vars = model.addVar(vtype="C", name="total_emission")

        # Objective: Maximize total profit
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        
        model.setObjective(objective_expr, "maximize")

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= z[i],
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )
        
        # Big M Constraints: Ensure z[i] logically connects to x[i][j]
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                model.addCons(var_names[(i, j)] <= z[i], f"BigM_constraint_1_{i}_{j}")  # If x[i][j] is 1, z[i] must be 1
                model.addCons(var_names[(i, j)] >= z[i] - (1 - var_names[(i, j)]), f"BigM_constraint_2_{i}_{j}")  # If z[i] is 1, at least one x[i][j] must be 1

        # Adding Clique Inequalities
        for clique_id, clique in enumerate(cliques):
            for j in range(number_of_knapsacks):
                model.addCons(quicksum(var_names[(i, j)] for i in clique) <= 1, f"Clique_{clique_id}_Knapsack_{j}")

        # New Constraints for Zone Assignments #
        for i in range(number_of_items):
            model.addCons(
                quicksum(y[(i, j)] for j in range(num_zones)) == z[i],
                f"ZoneAssignment_{i}"
            )

        for j in range(num_zones):
            model.addCons(
                quicksum(weights[i] * y[(i, j)] for i in range(number_of_items)) <= zone_capacities[j],
                f"ZoneCapacity_{j}"
            )

        # Emission Constraint
        model.addCons(
            emission_vars == quicksum(emissions[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)),
            "TotalEmission"
        )
        
        model.addCons(
            emission_vars <= sustainability_budget,
            "SustainabilityLimit"
        )
        
        # Fuel Consumption Constraint
        total_fuel_consumption = quicksum(fuel_consumption[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        model.addCons(
            total_fuel_consumption <= self.max_fuel_consumption,
            "FuelConsumptionLimit"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 150,
        'number_of_knapsacks': 25,
        'min_range': 20,
        'max_range': 56,
        'weight_mean': 37,
        'weight_std': 2500,
        'profit_mean_shift': 25,
        'profit_std': 40,
        'num_cliques': 22,
        'max_clique_size': 10,
        'min_zones': 50,
        'max_zones': 100,
        'min_emission': 0.1,
        'max_emission': 2.0,
        'min_zone_capacity': 1000,
        'max_zone_capacity': 5000,
        'min_fuel': 1.0,
        'max_fuel': 15.0,
        'sustainability_budget': 300,
        'max_fuel_consumption': 1500,
    }

    knapsack = MultipleKnapsack(parameters, seed=seed)
    instance = knapsack.generate_instance()
    solve_status, solve_time = knapsack.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")