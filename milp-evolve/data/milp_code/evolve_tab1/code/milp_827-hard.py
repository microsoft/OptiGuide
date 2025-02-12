import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from itertools import combinations


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

        # Neighborhood constraints setup
        neighborhood_size = 5
        neighborhoods = [list(range(i, min(i+neighborhood_size, self.number_of_centers))) for i in range(0, self.number_of_centers, neighborhood_size)]

        # Service probability
        service_probability = np.random.uniform(0.8, 1, self.number_of_centers)

        # Minimum flow threshold
        min_flow_threshold = np.random.randint(100, 300, size=(self.number_of_centers, self.number_of_customers))
        
        # Max flow per center
        max_flow_capacity = np.random.randint(1500, 3000, self.number_of_centers)
        
        res = {
            'setup_cost': setup_cost,
            'delivery_costs': delivery_costs,
            'meal_capacities': meal_capacities,
            'truck_capacities': truck_capacities,
            'delivery_feasibility': delivery_feasibility,
            'neighborhoods': neighborhoods,
            'service_probability': service_probability,
            'min_flow_threshold': min_flow_threshold,
            'max_flow_capacity': max_flow_capacity
        }

        return res

    ################# PySCIPOpt Modeling #################

    def solve(self, instance):
        setup_cost = instance['setup_cost']
        delivery_costs = instance['delivery_costs']
        meal_capacities = instance['meal_capacities']
        truck_capacities = instance['truck_capacities']
        delivery_feasibility = instance['delivery_feasibility']
        neighborhoods = instance['neighborhoods']
        service_probability = instance['service_probability']
        min_flow_threshold = instance['min_flow_threshold']
        max_flow_capacity = instance['max_flow_capacity']

        number_of_centers = len(setup_cost)
        number_of_customers = delivery_costs.shape[1]

        model = Model("NutritionDeliveryProblem")

        # Decision variables
        center_setup = {i: model.addVar(vtype="B", name=f"center_setup_{i}") for i in range(number_of_centers)}
        delivery_assignment = {(i, j): model.addVar(vtype="B", name=f"delivery_assignment_{i}_{j}") for i in range(number_of_centers) for j in range(number_of_customers)}
        flow_vars = {(i, j): model.addVar(vtype="C", name=f"flow_{i}_{j}") for i in range(number_of_centers) for j in range(number_of_customers)}

        # Objective: Minimize total cost (setup costs + delivery costs)
        objective_expr = quicksum(setup_cost[i] * center_setup[i] for i in range(number_of_centers))
        objective_expr += quicksum(delivery_costs[i][j] * delivery_assignment[(i, j)] for i in range(number_of_centers) for j in range(number_of_customers))
        
        # Adding service probability as an additional objective to maximize
        objective_expr -= quicksum(service_probability[i] * center_setup[i] for i in range(number_of_centers))

        ### given constraints and variables and objective code ends here
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

        # Constraint: Neighborhood constraints
        for neighborhood in neighborhoods:
            model.addCons(quicksum(center_setup[i] for i in neighborhood) <= 1, name=f"NeighborhoodConstraint_{neighborhood}")

        # Constraint: Flow constraints
        for i in range(number_of_centers):
            model.addCons(quicksum(flow_vars[(i, j)] for j in range(number_of_customers)) <= max_flow_capacity[i], f"FlowCapacity_{i}")

        for i in range(number_of_centers):
            for j in range(number_of_customers):
                model.addCons(flow_vars[(i, j)] >= min_flow_threshold[i, j] * delivery_assignment[(i, j)], name=f"MinFlow_{i}_{j}")
                model.addCons(flow_vars[(i, j)] <= delivery_assignment[(i, j)] * 1e5, name=f"MaxFlowEnablement_{i}_{j}")

        ### new constraints and variables and objective code ends here

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
        'max_delivery_cost': 600,
        'min_meal_capacity': 20,
        'max_meal_capacity': 2100,
        'min_truck_capacity': 450,
        'max_truck_capacity': 600,
        'max_delivery_range': 0.73,
    }
    ### given parameter code ends here
    ### new parameter code ends here
    
    nutrition_delivery = NutritionDeliveryProblem(parameters, seed=seed)
    instance = nutrition_delivery.generate_instance()
    solve_status, solve_time = nutrition_delivery.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")