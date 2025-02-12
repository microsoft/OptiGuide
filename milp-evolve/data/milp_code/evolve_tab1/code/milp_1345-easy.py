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

    ################# data generation #################
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
        
        # Diet related data generation
        calorie_content = np.random.randint(200, 800, size=(self.Number_of_Trucks, self.Number_of_Routes))
        nutrition_matrix = np.random.randint(0, 20, size=(self.Number_of_Trucks, self.Number_of_Routes, self.n_nutrients))
        daily_requirements = np.random.randint(50, 200, size=self.n_nutrients)
        
        # Enhanced Nutritional Constraints: Double-sided bounds on nutrients
        max_nutrient_bounds = np.random.randint(200, 500, size=self.n_nutrients)
        
        # Data related to dynamic route dependency constraints
        set_A = [0, 1, 2]  # Example routes
        set_B = [3, 4]  # Dependent routes
        
        # Data related to exclusive truck use conditions
        exclusive_truck_pairs = [(0, 1), (2, 3)]
        exclusive_route_pairs = [(0, 1), (2, 3)]
        
        return {
            "truck_costs": truck_costs,
            "route_costs": route_costs,
            "truck_capacities": truck_capacities,
            "route_demands": route_demands,
            "graph": graph,
            "incompatibilities": incompatibilities,
            "calorie_content": calorie_content,
            "nutrition_matrix": nutrition_matrix,
            "daily_requirements": daily_requirements,
            "max_nutrient_bounds": max_nutrient_bounds,
            "set_A": set_A,
            "set_B": set_B,
            "exclusive_truck_pairs": exclusive_truck_pairs,
            "exclusive_route_pairs": exclusive_route_pairs,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        truck_costs = instance['truck_costs']
        route_costs = instance['route_costs']
        truck_capacities = instance['truck_capacities']
        route_demands = instance['route_demands']
        graph = instance['graph']
        incompatibilities = instance['incompatibilities']
        calorie_content = instance['calorie_content']
        nutrition_matrix = instance['nutrition_matrix']
        daily_requirements = instance['daily_requirements']
        max_nutrient_bounds = instance['max_nutrient_bounds']
        set_A = instance['set_A']
        set_B = instance['set_B']
        exclusive_truck_pairs = instance['exclusive_truck_pairs']
        exclusive_route_pairs = instance['exclusive_route_pairs']
        
        model = Model("FleetManagement")
        number_of_trucks = len(truck_costs)
        number_of_routes = len(route_costs[0])
        n_nutrients = self.n_nutrients
        max_deviation = self.max_deviation
        target_calories = self.target_calories

        M = sum(truck_capacities)  # Big M
        
        # Decision variables
        truck_vars = {t: model.addVar(vtype="B", name=f"Truck_{t}") for t in range(number_of_trucks)}
        route_vars = {(t, r): model.addVar(vtype="B", name=f"Truck_{t}_Route_{r}") for t in range(number_of_trucks) for r in range(number_of_routes)}
        
        # Objective: minimize the total cost including truck startup costs and route service costs
        model.setObjective(
            quicksum(truck_costs[t] * truck_vars[t] for t in range(number_of_trucks)) +
            quicksum(route_costs[t, r] * route_vars[t, r] for t in range(number_of_trucks) for r in range(number_of_routes)), "minimize"
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

        # Nutritional constraints
        for n in range(n_nutrients):
            model.addCons(
                quicksum(route_vars[t, r] * nutrition_matrix[t, r, n] for t in range(number_of_trucks) for r in range(number_of_routes)) >= daily_requirements[n],
                f"NutritionalConstraints_{n}"
            )
        
        # Objective modification for deviation in calorie content
        total_calories = quicksum(route_vars[t, r] * calorie_content[t, r] for t in range(number_of_trucks) for r in range(number_of_routes))
        deviation = model.addVar(vtype="C", name="Deviation")
        model.addCons(total_calories - target_calories <= deviation, name="CaloricDeviationUpper")
        model.addCons(target_calories - total_calories <= deviation, name="CaloricDeviationLower")

        ### New logical constraints ###
        
        # Dynamic Route Dependency Constraints: If a route in set_A is served, a route in set_B must be too
        for r_a in set_A:
            for r_b in set_B:
                model.addCons(quicksum(route_vars[t, r_a] for t in range(number_of_trucks)) <= quicksum(route_vars[t, r_b] for t in range(number_of_trucks)), f"RouteDep_{r_a}_{r_b}")
        
        # Exclusive Truck Use Conditions: Trucks (i, j) cannot serve routes (r1, r2) together
        for (i, j), (r1, r2) in zip(exclusive_truck_pairs, exclusive_route_pairs):
            for t in range(number_of_trucks):
                model.addCons(truck_vars[i] <= 1 - route_vars[i, r1], f"Exclusive_{i}_{r1}")
                model.addCons(truck_vars[j] <= 1 - route_vars[j, r2], f"Exclusive_{j}_{r2}")

        # Enhanced Nutritional Constraints: Double-sided bounds on nutrients
        for n in range(n_nutrients):
            model.addCons(
                quicksum(route_vars[t, r] * nutrition_matrix[t, r, n] for t in range(number_of_trucks) for r in range(number_of_routes)) <= max_nutrient_bounds[n],
                f"MaxNutritionalConstraints_{n}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Trucks': 10,
        'Number_of_Routes': 22,
        'Route_Cost_Lower_Bound': 1200,
        'Route_Cost_Upper_Bound': 3000,
        'Min_Truck_Cost': 900,
        'Max_Truck_Cost': 5000,
        'Min_Truck_Capacity': 240,
        'Max_Truck_Capacity': 2000,
        'Affinity': 1,
        'n_nutrients': 100,
        'max_deviation': 1125,
        'target_calories': 100000,
    }

    fleet_management_optimizer = FleetManagement(parameters, seed)
    instance = fleet_management_optimizer.generate_instance()
    solve_status, solve_time, objective_value = fleet_management_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")