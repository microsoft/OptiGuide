import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WarehouseLayoutOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_units > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_facility_space > 0 and self.max_facility_space >= self.min_facility_space

        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_units))
        spaces = np.random.randint(self.min_facility_space, self.max_facility_space + 1, self.n_facilities)
        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_units).astype(int)

        G = nx.erdos_renyi_graph(self.n_facilities, self.graph_density, seed=self.seed)
        travel_times = {(u, v): np.random.randint(10, 60) for u, v in G.edges}
        travel_costs = {(u, v): np.random.uniform(10.0, 60.0) for u, v in G.edges}
        hotel_costs = np.random.randint(self.min_hotel_cost, self.max_hotel_cost + 1, self.n_facilities)
        hotel_availability = {node: np.random.uniform(50, 150) for node in G.nodes}
        
        return {
            "facility_costs": facility_costs,
            "transport_costs": transport_costs,
            "spaces": spaces,
            "demands": demands,
            "graph": G,
            "travel_times": travel_times,
            "travel_costs": travel_costs,
            "hotel_costs": hotel_costs,
            "hotel_availability": hotel_availability
        }
    
    # MILP modeling
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        transport_costs = instance['transport_costs']
        spaces = instance['spaces']
        demands = instance['demands']
        G = instance['graph']
        travel_times = instance['travel_times']
        travel_costs = instance['travel_costs']
        hotel_costs = instance['hotel_costs']
        hotel_availability = instance['hotel_availability']

        model = Model("WarehouseLayoutOptimization")
        n_facilities = len(facility_costs)
        n_units = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        route_vars = {(u, v): model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        hotel_vars = {node: model.addVar(vtype="C", name=f"Hotel_{node}") for node in G.nodes}
        
        # Objective: minimize the total cost (facility + transport + hotel + travel)
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f, u] * transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)) +
            quicksum(hotel_costs[f] * hotel_vars[f] for f in range(n_facilities)) +
            quicksum(travel_costs[(u, v)] * route_vars[(u, v)] for u, v in G.edges), 
            "minimize"
        )
        
        # Constraints: Each unit demand should be met by at least one facility
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) >= 1, f"Unit_{u}_Demand")
        
        # Constraints: Only open facilities can serve units
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f], f"Facility_{f}_Serve_{u}")
        
        # Constraints: Facilities cannot exceed their space
        for f in range(n_facilities):
            model.addCons(quicksum(demands[u] * transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")
        
        # Constraints: Only open facilities can route edges
        for u, v in G.edges:
            model.addCons(route_vars[(u, v)] <= facility_vars[u], f"Route_{u}_{v}_facility_{u}")
            model.addCons(route_vars[(u, v)] <= facility_vars[v], f"Route_{u}_{v}_facility_{v}")

        # Constraints: Hotel availability
        for node in G.nodes:
            model.addCons(hotel_vars[node] <= hotel_availability[node], name=f"Hotel_Limit_{node}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == 'optimal':
            objective_value = model.getObjVal()
        else:
            objective_value = None
        
        return model.getStatus(), end_time - start_time, objective_value

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 27,
        'n_units': 157,
        'min_transport_cost': 540,
        'max_transport_cost': 2400,
        'min_facility_cost': 2500,
        'max_facility_cost': 5000,
        'min_facility_space': 741,
        'max_facility_space': 1800,
        'mean_demand': 150,
        'std_dev_demand': 10,
        'graph_density': 0.38,
        'min_hotel_cost': 250,
        'max_hotel_cost': 2000,
    }

    warehouse_layout_optimizer = WarehouseLayoutOptimization(parameters, seed=42)
    instance = warehouse_layout_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_layout_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")