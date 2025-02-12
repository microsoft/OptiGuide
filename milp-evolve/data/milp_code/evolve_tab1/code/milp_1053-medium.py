import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class AirlineSchedulingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_aircraft > 0 and self.n_trips > 0
        assert self.min_cost_aircraft >= 0 and self.max_cost_aircraft >= self.min_cost_aircraft
        assert self.min_cost_travel >= 0 and self.max_cost_travel >= self.min_cost_travel
        assert self.min_capacity_aircraft > 0 and self.max_capacity_aircraft >= self.min_capacity_aircraft
        assert self.min_trip_demand >= 0 and self.max_trip_demand >= self.min_trip_demand

        aircraft_usage_costs = np.random.randint(self.min_cost_aircraft, self.max_cost_aircraft + 1, self.n_aircraft)
        travel_costs = np.random.randint(self.min_cost_travel, self.max_cost_travel + 1, (self.n_aircraft, self.n_trips))
        capacities = np.random.randint(self.min_capacity_aircraft, self.max_capacity_aircraft + 1, self.n_aircraft)
        trip_demands = np.random.randint(self.min_trip_demand, self.max_trip_demand + 1, self.n_trips)
        no_flight_penalties = np.random.uniform(100, 300, self.n_trips).tolist()

        return {
            "aircraft_usage_costs": aircraft_usage_costs,
            "travel_costs": travel_costs,
            "capacities": capacities,
            "trip_demands": trip_demands,
            "no_flight_penalties": no_flight_penalties,
        }

    def solve(self, instance):
        aircraft_usage_costs = instance['aircraft_usage_costs']
        travel_costs = instance['travel_costs']
        capacities = instance['capacities']
        trip_demands = instance['trip_demands']
        no_flight_penalties = instance['no_flight_penalties']

        model = Model("AirlineSchedulingOptimization")
        n_aircraft = len(aircraft_usage_costs)
        n_trips = len(trip_demands)
        
        aircraft_vars = {a: model.addVar(vtype="B", name=f"Aircraft_{a}") for a in range(n_aircraft)}
        trip_assignment_vars = {(a, t): model.addVar(vtype="C", name=f"Trip_{a}_Trip_{t}") for a in range(n_aircraft) for t in range(n_trips)}
        unmet_trip_vars = {t: model.addVar(vtype="C", name=f"Unmet_Trip_{t}") for t in range(n_trips)}

        # Objective function: Minimize total cost (aircraft usage + travel cost + penalty for unmet trips)
        model.setObjective(
            quicksum(aircraft_usage_costs[a] * aircraft_vars[a] for a in range(n_aircraft)) +
            quicksum(travel_costs[a][t] * trip_assignment_vars[a, t] for a in range(n_aircraft) for t in range(n_trips)) +
            quicksum(no_flight_penalties[t] * unmet_trip_vars[t] for t in range(n_trips)),
            "minimize"
        )

        # Constraints
        # Trip demand satisfaction (total flights and unmet trips must cover total demand)
        for t in range(n_trips):
            model.addCons(quicksum(trip_assignment_vars[a, t] for a in range(n_aircraft)) + unmet_trip_vars[t] == trip_demands[t], f"Trip_Demand_Satisfaction_{t}")
        
        # Capacity limits for each aircraft
        for a in range(n_aircraft):
            model.addCons(quicksum(trip_assignment_vars[a, t] for t in range(n_trips)) <= capacities[a] * aircraft_vars[a], f"Aircraft_Capacity_{a}")

        # Trip assignment only if aircraft is operational
        for a in range(n_aircraft):
            for t in range(n_trips):
                model.addCons(trip_assignment_vars[a, t] <= trip_demands[t] * aircraft_vars[a], f"Operational_Constraint_{a}_{t}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_aircraft': 200,
        'n_trips': 600,
        'min_cost_aircraft': 5000,
        'max_cost_aircraft': 10000,
        'min_cost_travel': 150,
        'max_cost_travel': 600,
        'min_capacity_aircraft': 1500,
        'max_capacity_aircraft': 2400,
        'min_trip_demand': 200,
        'max_trip_demand': 1000,
    }

    airline_optimizer = AirlineSchedulingOptimization(parameters, seed=42)
    instance = airline_optimizer.generate_instance()
    solve_status, solve_time, objective_value = airline_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")