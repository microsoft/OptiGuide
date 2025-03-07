import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class AdvancedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_transportation_costs(self, graph, node_positions, customers, facilities):
        m = len(customers)
        n = len(facilities)
        costs = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                customer_node = customers[i]
                facility_node = facilities[j]
                path_length = nx.shortest_path_length(graph, source=customer_node, target=facility_node, weight='weight')
                costs[i, j] = path_length
        return costs

    def generate_instance(self):
        # Generate customers and facility nodes in a random graph
        graph = nx.random_geometric_graph(self.n_nodes, radius=self.radius)
        pos = nx.get_node_attributes(graph, 'pos')
        customers = random.sample(list(graph.nodes), self.n_customers)
        facilities = random.sample(list(graph.nodes), self.n_facilities)
        
        demands = np.random.poisson(self.avg_demand, self.n_customers)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.randint(self.n_facilities, self.fixed_cost_interval)
        transportation_costs = self.generate_transportation_costs(graph, pos, customers, facilities)

        vehicle_capacities = self.randint(self.n_vehicle_types, self.vehicle_capacity_interval)
        vehicle_costs_per_km = np.random.uniform(self.vehicle_cost_min, self.vehicle_cost_max, self.n_vehicle_types)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'vehicle_capacities': vehicle_capacities,
            'vehicle_costs_per_km': vehicle_costs_per_km
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        vehicle_capacities = instance['vehicle_capacities']
        vehicle_costs_per_km = instance['vehicle_costs_per_km']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        n_vehicle_types = len(vehicle_capacities)
        
        model = Model("AdvancedFacilityLocation")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="I", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        vehicle_assign = {(j, k): model.addVar(vtype="B", name=f"VehicleAssign_{j}_{k}") for j in range(n_facilities) for k in range(n_vehicle_types)}
        unmet_demand = {i: model.addVar(vtype="I", name=f"UnmetDemand_{i}") for i in range(n_customers)}
        
        # Semi-Continuous Variable for minimum operational level
        min_operational_service = 100  # arbitrary lower bound for semi-continuous variable
        serve_continuous = {j: model.addVar(lb=min_operational_service, name=f"ServeContinuous_{j}") for j in range(n_facilities)}

        # Objective: minimize total cost including penalties for unmet demand
        fixed_costs_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities))
        transportation_costs_expr = quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        vehicle_costs_expr = quicksum(vehicle_costs_per_km[k] * vehicle_assign[j, k] * capacities[j] for j in range(n_facilities) for k in range(n_vehicle_types))
        unmet_demand_penalty = 1000  # Arbitrary large penalty for unmet demand

        objective_expr = fixed_costs_expr + transportation_costs_expr + vehicle_costs_expr + quicksum(unmet_demand_penalty * unmet_demand[i] for i in range(n_customers))

        # Constraints: demand must be met or penalized
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) + unmet_demand[i] == demands[i], f"Demand_{i}")
        
        # Constraints: capacity limits
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")
        
        # Constraints: vehicle capacity limits
        for j in range(n_facilities):
            model.addCons(quicksum(vehicle_assign[j, k] * vehicle_capacities[k] for k in range(n_vehicle_types)) >= capacities[j] * open_facilities[j], f"VehicleCapacity_{j}")

        # Constraints: tightening constraints
        for j in range(n_facilities):
            for i in range(n_customers):
                model.addCons(serve[i, j] <= open_facilities[j] * demands[i], f"Tightening_{i}_{j}")
        
        # New constraints: minimum operational service if open
        for j in range(n_facilities):
            model.addCons(serve_continuous[j] * open_facilities[j] <= capacities[j], f"MinOperationalService_{j}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 37,
        'n_facilities': 75,
        'n_nodes': 112,
        'radius': 0.8,
        'avg_demand': 35,
        'capacity_interval': (150, 2415),
        'fixed_cost_interval': (500, 555),
        'vehicle_capacity_interval': (750, 1500),
        'vehicle_cost_min': 13.5,
        'vehicle_cost_max': 5.0,
        'n_vehicle_types': 9,
        'min_operational_service': 100  # new parameter
    }
    
    facility_location = AdvancedFacilityLocation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")