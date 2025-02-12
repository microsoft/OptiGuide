import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class VehicleAllocationMILP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.zone_prob, directed=True, seed=self.seed)
        return G
    
    def generate_transport_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(10, 200)

        for u, v in G.edges:
            G[u][v]['distance'] = np.random.randint(1, 10)
            G[u][v]['capacity'] = np.random.randint(5, 20)
    
    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid
    
    def create_zones(self, G):
        zones = list(nx.find_cliques(G.to_undirected()))
        return zones
    
    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_transport_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        zones = self.create_zones(G)

        charging_capacity = {node: np.random.randint(20, 100) for node in G.nodes}
        trip_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}
        
        renewable_capacity = {node: np.random.randint(10, 50) for node in G.nodes}
        
        return {
            'G': G,
            'E_invalid': E_invalid, 
            'zones': zones, 
            'charging_capacity': charging_capacity, 
            'trip_cost': trip_cost, 
            'renewable_capacity': renewable_capacity
        }
    
    def solve(self, instance):
        G, E_invalid, zones = instance['G'], instance['E_invalid'], instance['zones']
        charging_capacity = instance['charging_capacity']
        trip_cost = instance['trip_cost']
        renewable_capacity = instance['renewable_capacity']

        model = Model("VehicleAllocation")
        
        # Define all variables
        vehicle_vars = {f"Vehicle_{node}": model.addVar(vtype="B", name=f"Vehicle_{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        shift_budget = model.addVar(vtype="C", name="shift_budget")

        # Define objective
        objective_expr = quicksum(
            G.nodes[node]['demand'] * vehicle_vars[f"Vehicle_{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['distance'] * route_vars[f"Route_{u}_{v}"]
            for u, v in E_invalid
        )

        objective_expr -= quicksum(
            trip_cost[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges
        )

        # Applying Convex Hull Formulation
        for i, zone in enumerate(zones):
            for j in range(len(zone)):
                for k in range(j + 1, len(zone)):
                    u, v = zone[j], zone[k]
                    model.addCons(
                        vehicle_vars[f"Vehicle_{u}"] + vehicle_vars[f"Vehicle_{v}"] <= 1,
                        name=f"ConvexHull_Zone_{i}_{u}_{v}"
                    )

        M = 1000  # Big M constant, set contextually larger than any decision boundary.

        for u, v in G.edges:
            # Convex Hull Constraints replacing Big-M
            model.addCons(
                vehicle_vars[f"Vehicle_{u}"] + vehicle_vars[f"Vehicle_{v}"] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"VehicleFlow_{u}_{v}"
            )
            model.addCons(
                vehicle_vars[f"Vehicle_{u}"] + vehicle_vars[f"Vehicle_{v}"] >= 2 * route_vars[f"Route_{u}_{v}"] - route_vars[f"Route_{u}_{v}"],
                name=f"VehicleFlow_{u}_{v}_other"
            )

        # Charging constraints
        for node in G.nodes:
            model.addCons(
                vehicle_vars[f"Vehicle_{node}"] <= charging_capacity[node],
                name=f"VehicleCharging_{node}"
            )

        model.addCons(
            shift_budget <= self.shift_hours,
            name="OffTime_Limit"
        )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 44,
        'max_nodes': 700,
        'zone_prob': 0.1,
        'exclusion_rate': 0.8,
        'shift_hours': 500,
        'no_of_scenarios': 2,
    }
    
    vehicle_allocation = VehicleAllocationMILP(parameters, seed=seed)
    instance = vehicle_allocation.get_instance()
    solve_status, solve_time = vehicle_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")