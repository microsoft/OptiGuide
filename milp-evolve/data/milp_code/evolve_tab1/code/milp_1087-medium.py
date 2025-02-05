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

        # Hubs and route coverage data
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        G, adj_mat, edge_list = self.generate_erdos_graph()
        coverage_costs, coverage_set = self.generate_coverage_data()
        
        return {
            "aircraft_usage_costs": aircraft_usage_costs,
            "travel_costs": travel_costs,
            "capacities": capacities,
            "trip_demands": trip_demands,
            "no_flight_penalties": no_flight_penalties,
            "adj_mat": adj_mat,
            "edge_list": edge_list,
            "coverage_costs": coverage_costs,
            "coverage_set": coverage_set
        }

    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))

        return G, adj_mat, edge_list
    
    def generate_coverage_data(self):
        coverage_costs = np.random.randint(1, self.coverage_max_cost + 1, size=self.n_nodes)
        coverage_pairs = [(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes) if i != j]
        chosen_pairs = np.random.choice(len(coverage_pairs), size=self.n_coverage_pairs, replace=False)
        coverage_set = [coverage_pairs[i] for i in chosen_pairs]
        return coverage_costs, coverage_set

    def solve(self, instance):
        aircraft_usage_costs = instance['aircraft_usage_costs']
        travel_costs = instance['travel_costs']
        capacities = instance['capacities']
        trip_demands = instance['trip_demands']
        no_flight_penalties = instance['no_flight_penalties']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        coverage_costs = instance['coverage_costs']
        coverage_set = instance['coverage_set']

        model = Model("AirlineSchedulingOptimization")
        n_aircraft = len(aircraft_usage_costs)
        n_trips = len(trip_demands)
        
        aircraft_vars = {a: model.addVar(vtype="B", name=f"Aircraft_{a}") for a in range(n_aircraft)}
        trip_assignment_vars = {(a, t): model.addVar(vtype="C", name=f"Trip_{a}_Trip_{t}") for a in range(n_aircraft) for t in range(n_trips)}
        unmet_trip_vars = {t: model.addVar(vtype="C", name=f"Unmet_Trip_{t}") for t in range(n_trips)}

        # New Variables: Route usage and coverage_vars
        route_vars = {(i,j): model.addVar(vtype="B", name=f"Route_{i}_{j}") for (i,j) in edge_list}
        coverage_vars = {(i, j): model.addVar(vtype="B", name=f"Coverage_{i}_{j}") for (i, j) in coverage_set}

        # New Objective Function: Minimize total cost (aircraft usage + travel cost + penalty for unmet trips + extra coverage and route costs)
        model.setObjective(
            quicksum(aircraft_usage_costs[a] * aircraft_vars[a] for a in range(n_aircraft)) +
            quicksum(travel_costs[a][t] * trip_assignment_vars[a, t] for a in range(n_aircraft) for t in range(n_trips)) +
            quicksum(no_flight_penalties[t] * unmet_trip_vars[t] for t in range(n_trips)) +
            quicksum(adj_mat[i, j][1] * route_vars[(i,j)] for (i,j) in edge_list) +
            quicksum(coverage_costs[i] * coverage_vars[(i, j)] for (i, j) in coverage_set),
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

        # Coverage constraints: ensure each trip passes through at least one hub
        for i, j in coverage_set:
            model.addCons(coverage_vars[(i, j)] == 1, f"Coverage_Requirement_{i}_{j}")
 
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
        'min_n_nodes': 20,
        'max_n_nodes': 30,
        'c_range': (11, 50),
        'd_range': (10, 100),
        'ratio': 100,
        'k_max': 10,
        'er_prob': 0.3,
        'n_coverage_pairs': 50,
        'coverage_max_cost': 20,
    }

    airline_optimizer = AirlineSchedulingOptimization(parameters, seed=seed)
    instance = airline_optimizer.generate_instance()
    solve_status, solve_time, objective_value = airline_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")