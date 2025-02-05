import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class LogisticsEmployeeAssignmentOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_vehicles > 0 and self.n_delivery_points > 0
        assert self.min_vehicle_cost >= 0 and self.max_vehicle_cost >= self.min_vehicle_cost
        assert self.min_fuel_cost >= 0 and self.max_fuel_cost >= self.min_fuel_cost
        assert self.min_vehicle_capacity > 0 and self.max_vehicle_capacity >= self.min_vehicle_capacity
        assert self.min_penalty_cost >= 0 and self.max_penalty_cost >= self.min_penalty_cost
        
        # Vehicle & Fuel costs
        vehicle_costs = np.random.randint(self.min_vehicle_cost, self.max_vehicle_cost + 1, self.n_vehicles)
        fuel_costs = np.random.randint(self.min_fuel_cost, self.max_fuel_cost + 1, (self.n_vehicles, self.n_delivery_points))
        
        # Capacities and demands
        vehicle_capacities = np.random.randint(self.min_vehicle_capacity, self.max_vehicle_capacity + 1, self.n_vehicles)
        delivery_demands = np.random.randint(1, 10, self.n_delivery_points)
        
        # Penalties and costs
        penalty_costs = np.random.uniform(self.min_penalty_cost, self.max_penalty_cost, self.n_delivery_points)
        salaries = np.random.uniform(self.min_salary, self.max_salary, self.n_employees)
        
        # New data
        miles_limits = np.random.uniform(self.min_miles_limit, self.max_miles_limit, self.n_vehicles)

        return {
            "vehicle_costs": vehicle_costs,
            "fuel_costs": fuel_costs,
            "vehicle_capacities": vehicle_capacities,
            "delivery_demands": delivery_demands,
            "penalty_costs": penalty_costs,
            "salaries": salaries,
            "miles_limits": miles_limits,
        }

    def solve(self, instance):
        vehicle_costs = instance['vehicle_costs']
        fuel_costs = instance['fuel_costs']
        vehicle_capacities = instance['vehicle_capacities']
        delivery_demands = instance['delivery_demands']
        penalty_costs = instance['penalty_costs']
        salaries = instance['salaries']
        miles_limits = instance['miles_limits']

        model = Model("LogisticsEmployeeAssignmentOptimization")
        n_vehicles = len(vehicle_costs)
        n_delivery_points = len(fuel_costs[0])
        n_employees = len(salaries)

        # Decision variables
        vehicle_vars = {v: model.addVar(vtype="B", name=f"Vehicle_{v}") for v in range(n_vehicles)}
        assignment_vars = {(v, d): model.addVar(vtype="C", name=f"Assignment_{v}_{d}") for v in range(n_vehicles) for d in range(n_delivery_points)}
        penalty_vars = {d: model.addVar(vtype="C", name=f"Penalty_{d}") for d in range(n_delivery_points)}
        employee_vars = {(v, e): model.addVar(vtype="B", name=f"Employee_{v}_{e}") for v in range(n_vehicles) for e in range(n_employees)}
        utilization_vars = {v: model.addVar(vtype="C", name=f"Utilization_{v}") for v in range(n_vehicles)}

        # Objective: minimize the total cost
        model.setObjective(
            quicksum(vehicle_costs[v] * vehicle_vars[v] for v in range(n_vehicles)) +
            quicksum(fuel_costs[v, d] * assignment_vars[(v, d)] for v in range(n_vehicles) for d in range(n_delivery_points)) +
            quicksum(penalty_costs[d] * penalty_vars[d] for d in range(n_delivery_points)) +
            quicksum(salaries[e] * employee_vars[(v, e)] for v in range(n_vehicles) for e in range(n_employees)),
            "minimize"
        )
        
        # Delivery balance constraints
        for d in range(n_delivery_points):
            model.addCons(
                quicksum(assignment_vars[(v, d)] for v in range(n_vehicles)) + penalty_vars[d] == delivery_demands[d], 
                f"Delivery_{d}_Balance"
            )

        # Vehicle capacity constraints
        for v in range(n_vehicles):
            model.addCons(
                quicksum(assignment_vars[(v, d)] for d in range(n_delivery_points)) <= vehicle_capacities[v] * vehicle_vars[v], 
                f"Vehicle_{v}_Capacity"
            )

        # Penalty variable non-negativity
        for d in range(n_delivery_points):
            model.addCons(penalty_vars[d] >= 0, f"Penalty_{d}_NonNegative")

        # Employee assignment constraints
        for v in range(n_vehicles):
            model.addCons(
                quicksum(employee_vars[(v, e)] for e in range(n_employees)) == vehicle_vars[v], 
                f"EmployeeAssignment_{v}"
            )

        # Mileage limits for vehicles
        for v in range(n_vehicles):
            model.addCons(utilization_vars[v] <= miles_limits[v], f"Vehicle_{v}_MileageLimit")
            model.addCons(
                quicksum(assignment_vars[(v, d)] for d in range(n_delivery_points)) == utilization_vars[v], 
                f"Vehicle_{v}_Utilization"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_vehicles': 50,
        'n_delivery_points': 100,
        'min_fuel_cost': 8,
        'max_fuel_cost': 250,
        'min_vehicle_cost': 2000,
        'max_vehicle_cost': 5000,
        'min_vehicle_capacity': 30,
        'max_vehicle_capacity': 100,
        'min_penalty_cost': 700,
        'max_penalty_cost': 3000,
        'min_salary': 3000,
        'max_salary': 6000,
        'min_miles_limit': 3000,
        'max_miles_limit': 5000,
        'n_employees': 80,
    }

    optimizer = LogisticsEmployeeAssignmentOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")