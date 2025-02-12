import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

class Fleet:
    def __init__(self, number_of_vehicles, vehicle_capacities, travel_times, demands):
        self.number_of_vehicles = number_of_vehicles
        self.vehicle_capacities = vehicle_capacities
        self.travel_times = travel_times
        self.demands = demands

class LogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.num_vehicles > 0 and self.num_tasks > 0
        assert self.min_capacity >= 0 and self.max_capacity >= self.min_capacity
        assert self.min_demand >= 0 and self.max_demand >= self.min_demand

        vehicle_capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.num_vehicles)
        demands = np.random.randint(self.min_demand, self.max_demand + 1, self.num_tasks)
        travel_times = np.random.randint(self.min_travel_time, self.max_travel_time + 1, (self.num_vehicles, self.num_tasks))
        operating_costs = np.random.randint(self.min_operating_cost, self.max_operating_cost + 1, self.num_vehicles)
        new_route_penalties = np.random.randint(self.min_new_route_penalty, self.max_new_route_penalty + 1, self.num_tasks)

        # Additional data from complex facility location
        energy_consumption = np.random.uniform(0.5, 2.0, self.num_vehicles).tolist()
        raw_material_availability = np.random.uniform(50, 200, self.num_tasks).tolist()
        labor_cost = np.random.uniform(10, 50, self.num_vehicles).tolist()
        environmental_impact = np.random.normal(20, 5, self.num_vehicles).tolist()
        
        return {
            "vehicle_capacities": vehicle_capacities,
            "demands": demands,
            "travel_times": travel_times,
            "operating_costs": operating_costs,
            "new_route_penalties": new_route_penalties,
            "energy_consumption": energy_consumption,
            "raw_material_availability": raw_material_availability,
            "labor_cost": labor_cost,
            "environmental_impact": environmental_impact
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        vehicle_capacities = instance['vehicle_capacities']
        demands = instance['demands']
        travel_times = instance['travel_times']
        operating_costs = instance['operating_costs']
        new_route_penalties = instance['new_route_penalties']
        energy_consumption = instance['energy_consumption']
        raw_material_availability = instance['raw_material_availability']
        labor_cost = instance['labor_cost']
        environmental_impact = instance['environmental_impact']
        
        model = Model("LogisticsOptimization")
        num_vehicles = len(vehicle_capacities)
        num_tasks = len(demands)

        # Decision variables
        Assignment_vars = {(v, t): model.addVar(vtype="B", name=f"Assignment_{v}_{t}") for v in range(num_vehicles) for t in range(num_tasks)}
        NewRoutes_vars = {t: model.addVar(vtype="B", name=f"NewRoute_{t}") for t in range(num_tasks)}
        TravelTime_vars = {(v, t): model.addVar(vtype="C", name=f"TravelTime_{v}_{t}") for v in range(num_vehicles) for t in range(num_tasks)}

        # New variables
        Energy_vars = {v: model.addVar(vtype="C", name=f"Energy_{v}", lb=0) for v in range(num_vehicles)}
        RawMaterial_vars = {t: model.addVar(vtype="C", name=f"RawMaterial_{t}", lb=0) for t in range(num_tasks)}
        LaborCost_vars = {v: model.addVar(vtype="C", name=f"LaborCost_{v}", lb=0) for v in range(num_vehicles)}
        EnvironmentalImpact_vars = {v: model.addVar(vtype="C", name=f"EnvironmentalImpact_{v}", lb=0) for v in range(num_vehicles)}

        # Objective: minimize the total travel time and operating costs
        model.setObjective(
            quicksum(travel_times[v, t] * Assignment_vars[v, t] for v in range(num_vehicles) for t in range(num_tasks)) +
            quicksum(operating_costs[v] * quicksum(Assignment_vars[v, t] for t in range(num_tasks)) for v in range(num_vehicles)) +
            quicksum(new_route_penalties[t] * NewRoutes_vars[t] for t in range(num_tasks)) +
            quicksum(Energy_vars[v] * energy_consumption[v] for v in range(num_vehicles)) +
            quicksum(LaborCost_vars[v] * labor_cost[v] for v in range(num_vehicles)) +
            quicksum(EnvironmentalImpact_vars[v] * environmental_impact[v] for v in range(num_vehicles)), 
            "minimize"
        )

        # Constraints: Each task must be assigned to exactly one vehicle or marked as a new route
        for t in range(num_tasks):
            model.addCons(quicksum(Assignment_vars[v, t] for v in range(num_vehicles)) + NewRoutes_vars[t] == 1, f"TaskAssignment_{t}")

        # Constraints: Vehicle capacity constraints
        for v in range(num_vehicles):
            model.addCons(quicksum(demands[t] * Assignment_vars[v, t] for t in range(num_tasks)) <= vehicle_capacities[v], f"VehicleCapacity_{v}")

        # Constraints: Travel time consistency
        M = self.M  # A large constant
        for v in range(num_vehicles):
            for t in range(num_tasks):
                model.addCons(TravelTime_vars[v, t] == travel_times[v, t] * Assignment_vars[v, t], f"TravelTimeConsistency_{v}_{t}")
                model.addCons(TravelTime_vars[v, t] <= M * Assignment_vars[v, t], f"BigM_TravelTime_{v}_{t}")

        # Additional constraints
        for v in range(num_vehicles):
            model.addCons(Energy_vars[v] == quicksum(Assignment_vars[v, t] * energy_consumption[v] for t in range(num_tasks)), f"EnergyConsumption_{v}")

        for t in range(num_tasks):
            model.addCons(RawMaterial_vars[t] <= raw_material_availability[t], f"RawMaterial_{t}")

        for v in range(num_vehicles):
            model.addCons(LaborCost_vars[v] <= labor_cost[v], f"LaborCost_{v}")

        for v in range(num_vehicles):
            model.addCons(EnvironmentalImpact_vars[v] <= environmental_impact[v], f"EnvironmentalImpact_{v}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_vehicles': 120,
        'num_tasks': 200,
        'min_capacity': 37,
        'max_capacity': 1800,
        'min_demand': 50,
        'max_demand': 1800,
        'min_travel_time': 3,
        'max_travel_time': 187,
        'min_operating_cost': 100,
        'max_operating_cost': 3000,
        'min_new_route_penalty': 500,
        'max_new_route_penalty': 600,
        'M': 2000,
    }

    logistics_optimizer = LogisticsOptimization(parameters, seed)
    instance = logistics_optimizer.generate_instance()
    solve_status, solve_time, objective_value = logistics_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")