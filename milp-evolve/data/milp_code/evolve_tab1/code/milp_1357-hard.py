import random
import time
import numpy as np
from pyscipopt import Model, quicksum

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

class ShippingCompanyOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.Number_of_Distribution_Centers > 0 and self.Number_of_Routes > 0
        assert self.Min_Center_Fixed_Cost >= 0 and self.Max_Center_Fixed_Cost >= self.Min_Center_Fixed_Cost
        assert self.Route_Delivery_Time_Lower_Bound >= 0 and self.Route_Delivery_Time_Upper_Bound >= self.Route_Delivery_Time_Lower_Bound
        assert self.Min_Center_Capacity > 0 and self.Max_Center_Capacity >= self.Min_Center_Capacity

        center_fixed_costs = np.random.randint(self.Min_Center_Fixed_Cost, self.Max_Center_Fixed_Cost + 1, self.Number_of_Distribution_Centers)
        route_delivery_times = np.random.randint(self.Route_Delivery_Time_Lower_Bound, self.Route_Delivery_Time_Upper_Bound + 1, (self.Number_of_Routes, self.Number_of_Distribution_Centers))
        center_capacities = np.random.randint(self.Min_Center_Capacity, self.Max_Center_Capacity + 1, self.Number_of_Distribution_Centers)
        route_traffic_congestion = np.random.uniform(1, self.Max_Traffic_Congestion, (self.Number_of_Routes, self.Number_of_Distribution_Centers))
        route_weather_impact = np.random.uniform(1, self.Max_Weather_Impact, (self.Number_of_Routes, self.Number_of_Distribution_Centers))
        route_carbon_emissions = np.random.uniform(self.Min_Route_Emissions, self.Max_Route_Emissions, (self.Number_of_Routes, self.Number_of_Distribution_Centers))

        traffic_delay_multiplier = np.random.uniform(1.0, 2.0, self.Number_of_Routes)

        return {
            "center_fixed_costs": center_fixed_costs,
            "route_delivery_times": route_delivery_times,
            "center_capacities": center_capacities,
            "route_traffic_congestion": route_traffic_congestion,
            "route_weather_impact": route_weather_impact,
            "route_carbon_emissions": route_carbon_emissions,
            "traffic_delay_multiplier": traffic_delay_multiplier
        }
        
    def solve(self, instance):
        center_fixed_costs = instance['center_fixed_costs']
        route_delivery_times = instance['route_delivery_times']
        center_capacities = instance['center_capacities']
        route_traffic_congestion = instance['route_traffic_congestion']
        route_weather_impact = instance['route_weather_impact']
        route_carbon_emissions = instance['route_carbon_emissions']
        traffic_delay_multiplier = instance['traffic_delay_multiplier']

        model = Model("ShippingCompanyOptimization")
        number_of_centers = len(center_fixed_costs)
        number_of_routes = len(route_delivery_times)

        # Decision variables
        center_vars = {c: model.addVar(vtype="B", name=f"Center_{c}") for c in range(number_of_centers)}
        route_vars = {(r, c): model.addVar(vtype="B", name=f"Route_{r}_Center_{c}") for r in range(number_of_routes) for c in range(number_of_centers)}
        carbon_vars = {(r, c): model.addVar(vtype="C", lb=0, name=f"Carbon_R_{r}_C_{c}") for r in range(number_of_routes) for c in range(number_of_centers)}

        # Objective: minimize the total operational cost including fixed costs, delivery time, congestion, and emissions
        model.setObjective(
            quicksum(center_fixed_costs[c] * center_vars[c] for c in range(number_of_centers)) +
            quicksum(route_delivery_times[r, c] * route_vars[r, c] * route_traffic_congestion[r, c] * traffic_delay_multiplier[r] * route_weather_impact[r, c] for r in range(number_of_routes) for c in range(number_of_centers)) +
            quicksum(carbon_vars[r, c] * route_carbon_emissions[r, c] for r in range(number_of_routes) for c in range(number_of_centers)), "minimize"
        )
        
        # Constraints: Each route demand is met by exactly one distribution center
        for r in range(number_of_routes):
            model.addCons(quicksum(route_vars[r, c] for c in range(number_of_centers)) == 1, f"Route_{r}_Demand")
        
        # Constraints: Only open distribution centers can be used for routes
        for c in range(number_of_centers):
            for r in range(number_of_routes):
                model.addCons(route_vars[r, c] <= center_vars[c], f"Center_{c}_Serve_{r}")

        # Constraints: Centers cannot exceed their capacity
        for c in range(number_of_centers):
            model.addCons(quicksum(route_vars[r, c] for r in range(number_of_routes)) <= center_capacities[c], f"Center_{c}_Capacity")

        # Constraints: Carbon emissions should be calculated based on route usage
        for r in range(number_of_routes):
            for c in range(number_of_centers):
                model.addCons(carbon_vars[r, c] == route_vars[r, c] * route_carbon_emissions[r, c], f"Carbon_Emission_{r}_{c}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Distribution_Centers': 10,
        'Number_of_Routes': 1000,
        'Route_Delivery_Time_Lower_Bound': 1,
        'Route_Delivery_Time_Upper_Bound': 5,
        'Min_Center_Fixed_Cost': 1000,
        'Max_Center_Fixed_Cost': 5000,
        'Min_Center_Capacity': 2,
        'Max_Center_Capacity': 1000,
        'Max_Traffic_Congestion': 2.0,
        'Max_Weather_Impact': 1.5,
        'Min_Route_Emissions': 7,
        'Max_Route_Emissions': 5,
    }

    shipping_optimizer = ShippingCompanyOptimization(parameters, seed=42)
    instance = shipping_optimizer.generate_instance()
    solve_status, solve_time, objective_value = shipping_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")