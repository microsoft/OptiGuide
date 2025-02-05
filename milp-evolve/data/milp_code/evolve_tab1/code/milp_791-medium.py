import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NutritionDeliveryProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################

    def generate_instance(self):
        # Generating setup costs for distribution centers
        setup_cost = np.random.randint(self.min_setup_cost, self.max_setup_cost, self.number_of_centers)
        
        # Generating operational delivery costs between centers and customer locations
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost, (self.number_of_centers, self.number_of_customers))
        
        # Generating meal handling capacities of distribution centers
        meal_capacities = np.random.randint(self.min_meal_capacity, self.max_meal_capacity, self.number_of_centers)

        # Generating truck capacities for meal delivery
        truck_capacities = np.random.randint(self.min_truck_capacity, self.max_truck_capacity, (self.number_of_centers, self.number_of_customers))

        # Delivery feasibility matrix
        distances = np.random.rand(self.number_of_centers, self.number_of_customers)
        delivery_feasibility = np.where(distances <= self.max_delivery_range, 1, 0)
        
        res = {
            'setup_cost': setup_cost,
            'delivery_costs': delivery_costs,
            'meal_capacities': meal_capacities,
            'truck_capacities': truck_capacities,
            'delivery_feasibility': delivery_feasibility,
        }
        return res

    ################# PySCIPOpt Modeling #################

    def solve(self, instance):
        setup_cost = instance['setup_cost']
        delivery_costs = instance['delivery_costs']
        meal_capacities = instance['meal_capacities']
        truck_capacities = instance['truck_capacities']
        delivery_feasibility = instance['delivery_feasibility']

        number_of_centers = len(setup_cost)
        number_of_customers = delivery_costs.shape[1]

        model = Model("NutritionDeliveryProblem")

        # Decision variables
        center_setup = {i: model.addVar(vtype="B", name=f"center_setup_{i}") for i in range(number_of_centers)}
        delivery_assignment = {(i, j): model.addVar(vtype="B", name=f"delivery_assignment_{i}_{j}") for i in range(number_of_centers) for j in range(number_of_customers)}

        # Objective: Minimize total cost (setup costs + delivery costs)
        objective_expr = quicksum(setup_cost[i] * center_setup[i] for i in range(number_of_centers))
        objective_expr += quicksum(delivery_costs[i][j] * delivery_assignment[(i, j)] for i in range(number_of_centers) for j in range(number_of_customers))

        model.setObjective(objective_expr, "minimize")

        # Constraint: Each customer must be serviced by at least one center within delivery range
        for j in range(number_of_customers):
            model.addCons(quicksum(center_setup[i] * delivery_feasibility[i][j] for i in range(number_of_centers)) >= 1, f"CustomerCoverage_{j}")

        # Constraint: Meal capacity constraints at distribution centers
        for i in range(number_of_centers):
            model.addCons(quicksum(delivery_assignment[(i, j)] for j in range(number_of_customers)) <= meal_capacities[i], f"MealCapacity_{i}")

        # Constraint: Truck capacity constraints
        for i in range(number_of_centers):
            for j in range(number_of_customers):
                model.addCons(delivery_assignment[(i, j)] <= truck_capacities[i][j], f"TruckCapacity_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_centers': 350,
        'number_of_customers': 300,
        'min_setup_cost': 2000,
        'max_setup_cost': 3000,
        'min_delivery_cost': 160,
        'max_delivery_cost': 800,
        'min_meal_capacity': 40,
        'max_meal_capacity': 2800,
        'min_truck_capacity': 450,
        'max_truck_capacity': 1200,
        'max_delivery_range': 0.75,
    }
    
    nutrition_delivery = NutritionDeliveryProblem(parameters, seed=seed)
    instance = nutrition_delivery.generate_instance()
    solve_status, solve_time = nutrition_delivery.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")