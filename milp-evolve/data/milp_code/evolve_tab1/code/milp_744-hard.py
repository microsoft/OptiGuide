import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CargoRoutingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_vehicles = random.randint(self.min_vehicles, self.max_vehicles)
        num_destinations = random.randint(self.min_destinations, self.max_destinations)

        # Cost matrices
        shipment_cost = np.random.randint(100, 500, size=(num_destinations, num_vehicles))
        operational_costs_vehicle = np.random.randint(2000, 5000, size=num_vehicles)

        # Destination demands
        destination_cargo = np.random.randint(200, 1000, size=num_destinations)

        # Vehicle capacities
        vehicle_capacity = np.random.randint(10000, 20000, size=num_vehicles)

        # Define breakpoints and slopes for heavy cargo penalty
        num_break_points = self.num_break_points
        break_points = np.sort(np.random.randint(0, self.max_capacity, size=(num_vehicles, num_break_points)))
        slopes = np.random.randint(5, 15, size=(num_vehicles, num_break_points + 1))

        res = {
            'num_vehicles': num_vehicles,
            'num_destinations': num_destinations,
            'shipment_cost': shipment_cost,
            'operational_costs_vehicle': operational_costs_vehicle,
            'destination_cargo': destination_cargo,
            'vehicle_capacity': vehicle_capacity,
            'break_points': break_points,
            'slopes': slopes,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_vehicles = instance['num_vehicles']
        num_destinations = instance['num_destinations']
        shipment_cost = instance['shipment_cost']
        operational_costs_vehicle = instance['operational_costs_vehicle']
        destination_cargo = instance['destination_cargo']
        vehicle_capacity = instance['vehicle_capacity']
        break_points = instance['break_points']
        slopes = instance['slopes']

        model = Model("CargoRoutingOptimization")

        # Variables
        Number_of_Vehicles = {v: model.addVar(vtype="B", name=f"Number_of_Vehicles_{v}") for v in range(num_vehicles)}
        Cargo_Shipment_Link = {(d, v): model.addVar(vtype="B", name=f"Cargo_Shipment_Link_{d}_{v}") for d in range(num_destinations) for v in range(num_vehicles)}

        # Auxiliary variables for heavy cargo penalties
        Maximum_Container_Usage = {v: model.addVar(vtype="C", name=f"Maximum_Container_Usage_{v}") for v in range(num_vehicles)}
        Heavy_Cargo_Limit = {(v, k): model.addVar(vtype="C", name=f"Heavy_Cargo_Limit_{v}_{k}") for v in range(num_vehicles) for k in range(len(break_points[0]) + 1)}
        
        # Objective function: Minimize total costs
        total_cost = quicksum(Cargo_Shipment_Link[d, v] * shipment_cost[d, v] for d in range(num_destinations) for v in range(num_vehicles)) + \
                     quicksum(Number_of_Vehicles[v] * operational_costs_vehicle[v] for v in range(num_vehicles)) + \
                     quicksum(Heavy_Cargo_Limit[v, k] for v in range(num_vehicles) for k in range(len(break_points[v]) + 1))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for d in range(num_destinations):
            model.addCons(quicksum(Cargo_Shipment_Link[d, v] for v in range(num_vehicles)) == 1, name=f"destination_delivery_{d}")

        for v in range(num_vehicles):
            for d in range(num_destinations):
                model.addCons(Cargo_Shipment_Link[d, v] <= Number_of_Vehicles[v], name=f"vehicle_shipment_link_{d}_{v}")

        for v in range(num_vehicles):
            model.addCons(quicksum(destination_cargo[d] * Cargo_Shipment_Link[d, v] for d in range(num_destinations)) == Maximum_Container_Usage[v], name=f"maximum_container_usage_{v}")
            for k in range(len(break_points[v]) + 1):
                if k == 0:
                    model.addCons(Heavy_Cargo_Limit[v, k] >= slopes[v, k] * quicksum(destination_cargo[d] * Cargo_Shipment_Link[d, v] for d in range(num_destinations)), name=f"heavy_cargo_limit_{v}_{k}")
                elif k == len(break_points[v]):
                    model.addCons(Heavy_Cargo_Limit[v, k] >= slopes[v, k] * (quicksum(destination_cargo[d] * Cargo_Shipment_Link[d, v] for d in range(num_destinations)) - break_points[v, k-1]), name=f"heavy_cargo_limit_{v}_{k}")
                else:
                    model.addCons(Heavy_Cargo_Limit[v, k] >= slopes[v, k] * (quicksum(destination_cargo[d] * Cargo_Shipment_Link[d, v] for d in range(num_destinations)) - break_points[v, k-1]) - Heavy_Cargo_Limit[v, k-1], name=f"heavy_cargo_limit_{v}_{k}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_vehicles': 150,
        'max_vehicles': 300,
        'min_destinations': 20,
        'max_destinations': 100,
        'num_break_points': 12,
        'max_capacity': 20000,
    }
    optimization = CargoRoutingOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")