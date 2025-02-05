import random
import time
import numpy as np
import networkx as nx
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
        
        critical_trips_subsets = [random.sample(range(self.n_trips), int(0.2 * self.n_trips)) for _ in range(5)]
        min_coverage = np.random.randint(1, 5, 5)
        
        # Battery life and weather disruption
        battery_life = np.random.randint(200, 500, self.n_aircraft)  # in minutes
        weather_disruptions = np.random.binomial(1, 0.1, self.n_trips)  # 10% disruption probability

        # Graph-based generation for piecewise constraints
        n_nodes = self.n_trips
        G = nx.erdos_renyi_graph(n=n_nodes, p=0.2, seed=self.seed)
        revenues = np.random.randint(50, 150, n_nodes)  # Individual trip revenues

        return {
            "aircraft_usage_costs": aircraft_usage_costs,
            "travel_costs": travel_costs,
            "capacities": capacities,
            "trip_demands": trip_demands,
            "no_flight_penalties": no_flight_penalties,
            "critical_trips_subsets": critical_trips_subsets,
            "min_coverage": min_coverage,
            "battery_life": battery_life,
            "weather_disruptions": weather_disruptions,
            "graph": G,
            "revenues": revenues
        }

    def solve(self, instance):
        aircraft_usage_costs = instance['aircraft_usage_costs']
        travel_costs = instance['travel_costs']
        capacities = instance['capacities']
        trip_demands = instance['trip_demands']
        no_flight_penalties = instance['no_flight_penalties']
        critical_trips_subsets = instance['critical_trips_subsets']
        min_coverage = instance['min_coverage']
        battery_life = instance['battery_life']
        weather_disruptions = instance['weather_disruptions']
        G = instance['graph']
        revenues = instance['revenues']

        model = Model("AirlineSchedulingOptimization")
        n_aircraft = len(aircraft_usage_costs)
        n_trips = len(trip_demands)
        
        aircraft_vars = {a: model.addVar(vtype="B", name=f"Aircraft_{a}") for a in range(n_aircraft)}
        trip_assignment_vars = {(a, t): model.addVar(vtype="C", name=f"Trip_{a}_{t}") for a in range(n_aircraft) for t in range(n_trips)}
        unmet_trip_vars = {t: model.addVar(vtype="C", name=f"Unmet_Trip_{t}") for t in range(n_trips)}
        delay_trip_vars = {t: model.addVar(vtype="C", name=f"Delayed_Trip_{t}") for t in range(n_trips)}
        
        # New piecewise linear variables for graph-based constraints
        piecewise_vars = {(u, v): model.addVar(vtype="B", name=f"Piecewise_{u}_{v}") for u, v in G.edges}

        model.setObjective(
            quicksum(aircraft_usage_costs[a] * aircraft_vars[a] for a in range(n_aircraft)) +
            quicksum(travel_costs[a][t] * (trip_assignment_vars[a, t] + delay_trip_vars[t]) for a in range(n_aircraft) for t in range(n_trips)) +
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

        # Set covering constraints: ensure minimum number of critical trips covered by some aircraft
        for i, subset in enumerate(critical_trips_subsets):
            model.addCons(quicksum(trip_assignment_vars[a, t] for a in range(n_aircraft) for t in subset) >= min_coverage[i], f"Set_Covering_Constraint_{i}")

        # New constraints for battery life
        for a in range(n_aircraft):
            model.addCons(quicksum(trip_assignment_vars[a, t] for t in range(n_trips)) <= battery_life[a], f"Battery_Life_{a}")

        # New constraints for weather disruptions
        for t in range(n_trips):
            if weather_disruptions[t] == 1:
                model.addCons(quicksum(trip_assignment_vars[a, t] for a in range(n_aircraft)) + unmet_trip_vars[t] + delay_trip_vars[t] >= trip_demands[t], f"Weather_Disruption_{t}")
        
        # Graph-based piecewise linear constraints
        for u, v in G.edges:
            model.addCons(
                quicksum(trip_assignment_vars[a, u] for a in range(n_aircraft)) + quicksum(trip_assignment_vars[a, v] for a in range(n_aircraft)) <= 1 + piecewise_vars[(u, v)],
                name=f"Graph_PL1_{u}_{v}"
            )
            model.addCons(
                quicksum(trip_assignment_vars[a, u] for a in range(n_aircraft)) + quicksum(trip_assignment_vars[a, v] for a in range(n_aircraft)) >= 2 * piecewise_vars[(u, v)],
                name=f"Graph_PL2_{u}_{v}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_aircraft': 375,
        'n_trips': 30,
        'min_cost_aircraft': 5000,
        'max_cost_aircraft': 10000,
        'min_cost_travel': 84,
        'max_cost_travel': 225,
        'min_capacity_aircraft': 281,
        'max_capacity_aircraft': 1800,
        'min_trip_demand': 3,
        'max_trip_demand': 312,
        'min_n': 12,
        'max_n': 32,
        'er_prob': 0.66,
    }

    airline_optimizer = AirlineSchedulingOptimization(parameters, seed=seed)
    instance = airline_optimizer.generate_instance()
    solve_status, solve_time, objective_value = airline_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")