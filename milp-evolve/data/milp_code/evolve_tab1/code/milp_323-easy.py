import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class AdvancedProductionPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_facilities = random.randint(self.min_facilities, self.max_facilities)
        n_products = random.randint(self.min_products, self.max_products)

        # Costs
        setup_costs = np.random.randint(1000, 10000, size=n_facilities)
        production_costs = np.random.randint(20, 200, size=(n_facilities, n_products))
        overtime_costs = np.random.randint(50, 500, size=(n_facilities, n_products))

        # Capacities and demands
        production_capacity = np.random.randint(50, 200, size=n_facilities)
        product_demand = np.random.randint(30, 150, size=n_products)
        maximum_overtime = np.random.randint(20, 100, size=n_facilities)

        res = {
            'n_facilities': n_facilities,
            'n_products': n_products,
            'setup_costs': setup_costs,
            'production_costs': production_costs,
            'overtime_costs': overtime_costs,
            'production_capacity': production_capacity,
            'product_demand': product_demand,
            'maximum_overtime': maximum_overtime
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_facilities = instance['n_facilities']
        n_products = instance['n_products']
        setup_costs = instance['setup_costs']
        production_costs = instance['production_costs']
        overtime_costs = instance['overtime_costs']
        production_capacity = instance['production_capacity']
        product_demand = instance['product_demand']
        maximum_overtime = instance['maximum_overtime']

        model = Model("AdvancedProductionPlanning")

        # Variables
        FacilitySetup = {i: model.addVar(vtype="B", name=f"FacilitySetup_{i}") for i in range(n_facilities)}
        NumberOfProductsToProduce = {(i, j): model.addVar(vtype="I", name=f"NumberOfProductsToProduce_{i}_{j}") for i in range(n_facilities) for j in range(n_products)}
        AlternativeProductionTime = {(i, j): model.addVar(vtype="B", name=f"AlternativeProductionTime_{i}_{j}") for i in range(n_facilities) for j in range(n_products)}

        # Objective function: Minimize total cost
        total_cost = (
            quicksum(FacilitySetup[i] * setup_costs[i] for i in range(n_facilities)) +
            quicksum(NumberOfProductsToProduce[i, j] * production_costs[i, j] for i in range(n_facilities) for j in range(n_products)) +
            quicksum(AlternativeProductionTime[i, j] * overtime_costs[i, j] for i in range(n_facilities) for j in range(n_products))
        )

        model.setObjective(total_cost, "minimize")

        # Constraints
        for j in range(n_products):
            model.addCons(quicksum(NumberOfProductsToProduce[i, j] for i in range(n_facilities)) == product_demand[j], name=f"product_demand_{j}")

        for i in range(n_facilities):
            model.addCons(
                quicksum(NumberOfProductsToProduce[i, j] for j in range(n_products)) <= production_capacity[i] * FacilitySetup[i],
                name=f"production_capacity_{i}"
            )

            model.addCons(
                quicksum(NumberOfProductsToProduce[i, j] for j in range(n_products)) + 
                quicksum(AlternativeProductionTime[i, j] * maximum_overtime[i] for j in range(n_products)) <= production_capacity[i] * FacilitySetup[i],
                name=f"total_capacity_with_overtime_{i}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facilities': 20,
        'max_facilities': 300,
        'min_products': 70,
        'max_products': 400,
        'penalty_coefficient': 0.3,
    }

    planning = AdvancedProductionPlanning(parameters, seed=seed)
    instance = planning.generate_instance()
    solve_status, solve_time = planning.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")