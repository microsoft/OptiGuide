import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FleetAssignmentProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_aircraft_fleet(self, num_aircraft, num_flights):
        fleet = {}
        for a in range(num_aircraft):
            fleet[f'Aircraft_{a}'] = {
                'available_hours': np.random.randint(self.min_hours, self.max_hours + 1),
                'hourly_cost': np.random.uniform(self.min_cost, self.max_cost)
            }
        flights = [f'Flight_{f}' for f in range(num_flights)]
        return fleet, flights

    def generate_flight_times(self, num_flights):
        return [np.random.randint(self.min_flight_time, self.max_flight_time + 1) for _ in range(num_flights)]

    def generate_instances(self):
        num_aircraft = np.random.randint(self.min_aircraft, self.max_aircraft + 1)
        num_flights = np.random.randint(self.min_flights, self.max_flights + 1)
        fleet, flights = self.generate_aircraft_fleet(num_aircraft, num_flights)
        flight_times = self.generate_flight_times(num_flights)

        res = {
            'fleet': fleet,
            'flights': flights,
            'flight_times': flight_times
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        fleet = instance['fleet']
        flights = instance['flights']
        flight_times = instance['flight_times']

        model = Model("FleetAssignmentProblem")
        var_names = {}  

        # Create variables for each aircraft-flight assignment
        for aircraft in fleet:
            for flight in flights:
                var_name = f"{aircraft}_{flight}"
                var_names[var_name] = model.addVar(vtype="B", name=var_name)

        # NumberOfFleetsUsed (auxiliary integer variable for number of fleets used)
        num_fleets_used = model.addVar(vtype="I", name="NumberOfFleetsUsed")

        # Objective function - minimize the total cost of aircraft assignment
        objective_expr = quicksum(
            var_names[f"{aircraft}_{flight}"] * fleet[aircraft]['hourly_cost'] * flight_times[idx]
            for idx, flight in enumerate(flights) for aircraft in fleet
        )
        
        # Aircraft usage constraints (ensuring aircraft available hours constraint)
        for aircraft in fleet:
            model.addCons(
                quicksum(var_names[f"{aircraft}_{flight}"] * flight_times[idx] for idx, flight in enumerate(flights))
                <= fleet[aircraft]['available_hours'], 
                name=f"AircraftUsageConstraint_{aircraft}"
            )
        
        # Ensure each flight is assigned exactly one aircraft
        for idx, flight in enumerate(flights):
            model.addCons(
                quicksum(var_names[f"{aircraft}_{flight}"] for aircraft in fleet) == 1,
                name=f"FlightCoverageConstraint_{flight}"
            )
        
        # Number of fleets used constraint
        model.addCons(num_fleets_used == quicksum(var_names[f"{aircraft}_{flight}"] for aircraft in fleet for flight in flights), 
                      name="TotalFleetsUsedConstraint")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
            
if __name__ == '__main__':
    seed = 1234
    parameters = {
        'min_aircraft': 30,
        'max_aircraft': 50,
        'min_flights': 40,
        'max_flights': 600,
        'min_hours': 300,
        'max_hours': 1500,
        'min_cost': 1500,
        'max_cost': 2000,
        'min_flight_time': 5,
        'max_flight_time': 40,
    }

    fleet_assignment = FleetAssignmentProblem(parameters, seed=seed)
    instance = fleet_assignment.generate_instances()
    solve_status, solve_time = fleet_assignment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")