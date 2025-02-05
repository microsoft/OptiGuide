import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class FleetManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.Number_of_Trucks > 0 and self.Number_of_Routes > 0
        assert self.Min_Truck_Cost >= 0 and self.Max_Truck_Cost >= self.Min_Truck_Cost
        assert self.Route_Cost_Lower_Bound >= 0 and self.Route_Cost_Upper_Bound >= self.Route_Cost_Lower_Bound
        assert self.Min_Truck_Capacity > 0 and self.Max_Truck_Capacity >= self.Min_Truck_Capacity

        truck_costs = np.random.randint(self.Min_Truck_Cost, self.Max_Truck_Cost + 1, self.Number_of_Trucks)
        route_costs = np.random.randint(self.Route_Cost_Lower_Bound, self.Route_Cost_Upper_Bound + 1, (self.Number_of_Trucks, self.Number_of_Routes))
        truck_capacities = np.random.randint(self.Min_Truck_Capacity, self.Max_Truck_Capacity + 1, self.Number_of_Trucks)
        route_demands = np.ones(self.Number_of_Routes) * 5  # Simplified to constant demand
        
        graph = Graph.barabasi_albert(self.Number_of_Trucks, self.Affinity)
        incompatibilities = set(graph.edges)
        
        # Data related to dynamic route dependency constraints
        set_A = [0, 1, 2]  # Example routes
        set_B = [3, 4]  # Dependent routes

        # New Data for Time Windows and Fuel Prices
        early_time_windows = np.random.randint(self.Min_Early_Time_Window, self.Max_Early_Time_Window, self.Number_of_Routes)
        late_time_windows = np.random.randint(self.Min_Late_Time_Window, self.Max_Late_Time_Window, self.Number_of_Routes)
        fuel_prices = np.random.uniform(self.Min_Fuel_Price, self.Max_Fuel_Price)
        
        # New Data for Big M Formulation constraints
        route_service_times = np.random.randint(self.Min_Service_Time, self.Max_Service_Time, self.Number_of_Routes)
        max_service_time = np.max(route_service_times)
        
        # New Data for energy, labor, and environmental impact
        energy_costs = np.random.uniform(100, 500, self.Number_of_Trucks)
        energy_limits = np.random.uniform(1000, 2000, self.Number_of_Trucks)
        labor_costs = np.random.uniform(15, 60, self.Number_of_Trucks)
        worker_availability = np.random.randint(5, 20, self.Number_of_Trucks)
        environmental_impact = np.random.uniform(10, 50, self.Number_of_Trucks)
        
        mutual_exclusivity_pairs = []
        for _ in range(10):  # Example mutual exclusivity constraints
            t1 = random.randint(0, self.Number_of_Trucks - 1)
            t2 = random.randint(0, self.Number_of_Trucks - 1)
            if t1 != t2:
                mutual_exclusivity_pairs.append((t1, t2))
        
        return {
            "truck_costs": truck_costs,
            "route_costs": route_costs,
            "truck_capacities": truck_capacities,
            "route_demands": route_demands,
            "graph": graph,
            "incompatibilities": incompatibilities,
            "set_A": set_A,
            "set_B": set_B,
            "early_time_windows": early_time_windows,
            "late_time_windows": late_time_windows,
            "fuel_prices": fuel_prices,
            "route_service_times": route_service_times,
            "max_service_time": max_service_time,
            "energy_costs": energy_costs,
            "energy_limits": energy_limits,
            "labor_costs": labor_costs,
            "worker_availability": worker_availability,
            "environmental_impact": environmental_impact,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs
        }
        
    def solve(self, instance):
        truck_costs = instance['truck_costs']
        route_costs = instance['route_costs']
        truck_capacities = instance['truck_capacities']
        route_demands = instance['route_demands']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        set_A = instance['set_A']
        set_B = instance['set_B']
        early_time_windows = instance['early_time_windows']
        late_time_windows = instance['late_time_windows']
        fuel_prices = instance['fuel_prices']
        route_service_times = instance['route_service_times']
        max_service_time = instance['max_service_time']
        energy_costs = instance['energy_costs']
        energy_limits = instance['energy_limits']
        labor_costs = instance['labor_costs']
        worker_availability = instance['worker_availability']
        environmental_impact = instance['environmental_impact']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']

        model = Model("FleetManagement")
        number_of_trucks = len(truck_costs)
        number_of_routes = len(route_costs[0])
        M = max_service_time  # Big M
        
        # Decision variables
        truck_vars = {t: model.addVar(vtype="B", name=f"Truck_{t}") for t in range(number_of_trucks)}
        route_vars = {(t, r): model.addVar(vtype="B", name=f"Truck_{t}_Route_{r}") for t in range(number_of_trucks) for r in range(number_of_routes)}
        delivery_time_vars = {r: model.addVar(vtype="C", name=f"Delivery_Time_{r}") for r in range(number_of_routes)}
        
        # New decision variables for energy, labor, and environmental impact
        energy_vars = {t: model.addVar(vtype="C", name=f"Energy_{t}", lb=0) for t in range(number_of_trucks)}
        labor_vars = {t: model.addVar(vtype="C", name=f"Labor_{t}", lb=0) for t in range(number_of_trucks)}
        environmental_vars = {t: model.addVar(vtype="C", name=f"Environment_{t}", lb=0) for t in range(number_of_trucks)}
        
        # Objective: minimize the total cost including truck startup costs, route service costs, energy, labor, and environmental impact
        model.setObjective(
            quicksum(truck_costs[t] * truck_vars[t] for t in range(number_of_trucks)) +
            quicksum(route_costs[t, r] * route_vars[t, r] * fuel_prices for t in range(number_of_trucks) for r in range(number_of_routes)) +
            quicksum(energy_costs[t] * energy_vars[t] for t in range(number_of_trucks)) +
            quicksum(labor_costs[t] * labor_vars[t] for t in range(number_of_trucks)) +
            quicksum(environmental_vars[t] * environmental_impact[t] for t in range(number_of_trucks)),
            "minimize"
        )
        
        # Constraints: Each route must be served by exactly one truck
        for r in range(number_of_routes):
            model.addCons(quicksum(route_vars[t, r] for t in range(number_of_trucks)) == 1, f"Route_{r}_Demand")
        
        # Constraints: Only active trucks can serve routes
        for t in range(number_of_trucks):
            for r in range(number_of_routes):
                model.addCons(route_vars[t, r] <= truck_vars[t], f"Truck_{t}_Serve_{r}")
        
        # Constraints: Trucks cannot exceed their capacity using Big M
        for t in range(number_of_trucks):
            model.addCons(quicksum(route_demands[r] * route_vars[t, r] for r in range(number_of_routes)) <= truck_capacities[t] * truck_vars[t], f"Truck_{t}_Capacity")

        # Constraints: Truck Graph Incompatibilities
        for count, (i, j) in enumerate(incompatibilities):
            model.addCons(truck_vars[i] + truck_vars[j] <= 1, f"Incompatibility_{count}")

        # Dynamic Route Dependency Constraints
        for r_a in set_A:
            for r_b in set_B:
                model.addCons(quicksum(route_vars[t, r_a] for t in range(number_of_trucks)) <= quicksum(route_vars[t, r_b] for t in range(number_of_trucks)), f"RouteDep_{r_a}_{r_b}")
        
        # Big M Formulation: Ensure a truck cannot serve exclusive routes
        for t in range(number_of_trucks):
            for r in set_A:
                for r_prime in set_B:
                    model.addCons(delivery_time_vars[r] - delivery_time_vars[r_prime] + (2 - route_vars[t, r] - route_vars[t, r_prime]) * M >= route_service_times[r], f"Exclusivity_BigM_{t}_{r}_{r_prime}")
                    model.addCons(delivery_time_vars[r_prime] - delivery_time_vars[r] + (2 - route_vars[t, r] - route_vars[t, r_prime]) * M >= route_service_times[r_prime], f"Exclusivity_BigM_{t}_{r_prime}_{r}")

        # Time Windows Constraints
        for r in range(number_of_routes):
            model.addCons(delivery_time_vars[r] >= early_time_windows[r], f"Early_Time_Window_{r}")
            model.addCons(delivery_time_vars[r] <= late_time_windows[r], f"Late_Time_Window_{r}")

        # New Constraints: Energy limits per truck
        for t in range(number_of_trucks):
            model.addCons(energy_vars[t] <= energy_limits[t], f"Energy_Limit_{t}")
        
        # New Constraints: Labor limits per truck
        for t in range(number_of_trucks):
            model.addCons(labor_vars[t] <= worker_availability[t], f"Labor_Availability_{t}")
        
        # New Constraints: Environmental impact limits
        for t in range(number_of_trucks):
            model.addCons(environmental_vars[t] <= environmental_impact[t], f"Environmental_Impact_{t}")
        
        # New Constraints: Mutual exclusivity between trucks
        for (t1, t2) in mutual_exclusivity_pairs:
            model.addCons(truck_vars[t1] + truck_vars[t2] <= 1, f"Mutual_Exclusivity_{t1}_{t2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Trucks': 30,
        'Number_of_Routes': 115,
        'Route_Cost_Lower_Bound': 900,
        'Route_Cost_Upper_Bound': 3000,
        'Min_Truck_Cost': 1800,
        'Max_Truck_Cost': 5000,
        'Min_Truck_Capacity': 360,
        'Max_Truck_Capacity': 1500,
        'Affinity': 1,
        'Min_Early_Time_Window': 0,
        'Max_Early_Time_Window': 2,
        'Min_Late_Time_Window': 7,
        'Max_Late_Time_Window': 20,
        'Min_Fuel_Price': 3.0,
        'Max_Fuel_Price': 0.75,
        'Min_Service_Time': 100,
        'Max_Service_Time': 150,
        'Energy_Min_Cost': 100,
        'Energy_Max_Cost': 500,
        'Energy_Min_Limit': 1000,
        'Energy_Max_Limit': 2000,
        'Labor_Min_Cost': 15,
        'Labor_Max_Cost': 60,
        'Labor_Min_Availability': 5,
        'Labor_Max_Availability': 20,
        'Environmental_Min_Impact': 10,
        'Environmental_Max_Impact': 50,
    }

    fleet_management_optimizer = FleetManagement(parameters, seed)
    instance = fleet_management_optimizer.generate_instance()
    solve_status, solve_time, objective_value = fleet_management_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")