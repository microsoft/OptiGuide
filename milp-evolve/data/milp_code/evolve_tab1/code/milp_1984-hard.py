import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DeliveryFleet:
    """Helper function: Container for a delivery fleet system."""
    def __init__(self, number_of_vehicles, deliveries, operational_costs):
        self.number_of_vehicles = number_of_vehicles
        self.deliveries = deliveries
        self.operational_costs = operational_costs

class EmployeeAssignmentOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    # Sample the delivery requirements and vehicle capacities array
    def generate_deliveries(self, number_of_deliveries):
        capacities = np.random.randint(20, 100, size=number_of_deliveries)
        demands = np.random.randint(10, 80, size=number_of_deliveries)
        skill_requirements = np.random.randint(0, 5, size=number_of_deliveries)  # Skill requirements for deliveries
        return capacities, demands, skill_requirements

    ################# Data Generation #################
    def generate_fleet(self):
        deliveries = list(range(self.n_deliveries))
        operational_costs = np.random.randint(200, 500, size=self.n_vehicles)  # Operational costs for each vehicle
        return DeliveryFleet(self.n_vehicles, deliveries, operational_costs)

    def generate_instance(self):
        capacities, demands, skill_requirements = self.generate_deliveries(self.n_deliveries)
        fleet = self.generate_fleet()
        vehicle_capacities = np.random.normal(300, 50, size=self.n_vehicles).astype(int)
        
        employee_costs = np.random.randint(50, 200, size=self.n_employees)
        employee_skills = np.random.randint(0, 5, size=self.n_employees)  # Skill levels of employees
        
        res = {
            'fleet': fleet,
            'capacities': capacities,
            'demands': demands,
            'skill_requirements': skill_requirements,
            'vehicle_capacities': vehicle_capacities,
            'employee_costs': employee_costs,
            'employee_skills': employee_skills,
        }
        return res

    ################## PySCIPOpt Modeling #################
    def solve(self, instance):
        fleet = instance['fleet']
        capacities = instance['capacities']
        demands = instance['demands']
        skill_requirements = instance['skill_requirements']
        vehicle_capacities = instance['vehicle_capacities']
        employee_costs = instance['employee_costs']
        employee_skills = instance['employee_skills']

        model = Model("EmployeeAssignmentOptimization")

        # Variables
        delivery_vars = {(v, d): model.addVar(vtype="B", name=f"Delivery_{v}_{d}") for v in range(self.n_vehicles) for d in fleet.deliveries}
        vehicle_vars = {v: model.addVar(vtype="B", name=f"Vehicle_{v}") for v in range(self.n_vehicles)}  # Binary var indicating vehicle usage
        employee_vars = {(e, d): model.addVar(vtype="B", name=f"Employee_{e}_{d}") for e in range(self.n_employees) for d in fleet.deliveries}
        
        # Constraints
        for v in range(self.n_vehicles):
            model.addCons(quicksum(delivery_vars[v, d] * demands[d] for d in fleet.deliveries) <= vehicle_capacities[v] * vehicle_vars[v], name=f"VehicleCapacity_{v}")

        for d in fleet.deliveries:
            model.addCons(quicksum(delivery_vars[v, d] for v in range(self.n_vehicles)) == 1, name=f"DeliveryAssignment_{d}")

        for d in fleet.deliveries:
            model.addCons(quicksum(employee_vars[e, d] for e in range(self.n_employees) if employee_skills[e] >= skill_requirements[d]) >= 1, name=f"EmployeeSkillMatch_{d}")

        for v in range(self.n_vehicles):
            model.addCons(quicksum(delivery_vars[v, d] for d in fleet.deliveries) <= self.big_m * vehicle_vars[v], name=f"VehicleUsage_{v}")

        # Objective Function
        model.setObjective(
            quicksum(delivery_vars[v, d] * demands[d] for v in range(self.n_vehicles) for d in fleet.deliveries) 
            + quicksum(employee_vars[e, d] * employee_costs[e] for e in range(self.n_employees) for d in fleet.deliveries)
            + quicksum(vehicle_vars[v] * fleet.operational_costs[v] for v in range(self.n_vehicles)), 
            "minimize"
        )
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_deliveries': 100,
        'n_vehicles': 30,
        'n_employees': 450,
        'big_m': 2250,
    }
    
    employee_assignment_optimization = EmployeeAssignmentOptimization(parameters, seed=seed)
    instance = employee_assignment_optimization.generate_instance()
    solve_status, solve_time = employee_assignment_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")