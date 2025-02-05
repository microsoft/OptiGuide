import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MedicalSupplyAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data Generation
    def generate_city_graph(self):
        G = nx.random_geometric_graph(self.n_total_nodes, self.geo_radius, seed=self.seed)
        adj_mat = np.zeros((self.n_total_nodes, self.n_total_nodes), dtype=object)
        edge_list = []
        distribution_center_capacities = [random.randint(1, self.max_center_capacity) for _ in range(self.n_centers)]
        location_demand = [(random.randint(1, self.max_demand), random.randint(1, self.max_urgency)) for _ in range(self.n_locations)]

        for i, j in G.edges:
            cost = np.random.uniform(*self.transport_cost_range)
            adj_mat[i, j] = cost
            edge_list.append((i, j))

        distribution_centers = range(self.n_centers)
        locations = range(self.n_locations, self.n_total_nodes)
        
        return G, adj_mat, edge_list, distribution_center_capacities, location_demand, distribution_centers, locations

    def generate_instance(self):
        self.n_total_nodes = self.n_centers + self.n_locations
        G, adj_mat, edge_list, distribution_center_capacities, location_demand, distribution_centers, locations = self.generate_city_graph()

        res = {
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'distribution_center_capacities': distribution_center_capacities,
            'location_demand': location_demand,
            'distribution_centers': distribution_centers,
            'locations': locations
        }
        
        return res
    
    # PySCIPOpt Modeling
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        distribution_center_capacities = instance['distribution_center_capacities']
        location_demand = instance['location_demand']
        distribution_centers = instance['distribution_centers']
        locations = instance['locations']
        
        model = Model("MedicalSupplyAllocation")
        y_vars = {f"New_Center_{c+1}": model.addVar(vtype="B", name=f"New_Center_{c+1}") for c in distribution_centers}
        x_vars = {f"Center_Assign_{c+1}_{l+1}": model.addVar(vtype="B", name=f"Center_Assign_{c+1}_{l+1}") for c in distribution_centers for l in locations}
        
        # Objective function: minimize the total transport and operation cost
        objective_expr = quicksum(
            adj_mat[c, l] * x_vars[f"Center_Assign_{c+1}_{l+1}"]
            for c in distribution_centers for l in locations if adj_mat[c, l] != 0
        )
        # Adding maintenance cost for opening a distribution center
        objective_expr += quicksum(
            self.operation_cost * y_vars[f"New_Center_{c+1}"]
            for c in distribution_centers
        )

        model.setObjective(objective_expr, "minimize")

        # Constraints
        # Each location must be served by at least one distribution center (Set Covering constraint)
        for l in locations:
            model.addCons(quicksum(x_vars[f"Center_Assign_{c+1}_{l+1}"] for c in distribution_centers) >= 1, f"Serve_{l+1}")

        # Distribution center should be open if it serves any location
        for c in distribution_centers:
            for l in locations:
                model.addCons(x_vars[f"Center_Assign_{c+1}_{l+1}"] <= y_vars[f"New_Center_{c+1}"], f"Open_Cond_{c+1}_{l+1}")

        # Distribution center capacity constraint
        for c in distribution_centers:
            model.addCons(quicksum(location_demand[l-self.n_locations][0] * x_vars[f"Center_Assign_{c+1}_{l+1}"] for l in locations) <= distribution_center_capacities[c], f"Capacity_{c+1}")

        # Urgency delivery constraints using convex hull formulation
        for l in locations:
            for c in distribution_centers:
                urgency = location_demand[l-self.n_locations][1]
                model.addCons(x_vars[f"Center_Assign_{c+1}_{l+1}"] * urgency <= self.max_urgency, f"Urgency_{c+1}_{l+1}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_centers': 60,
        'n_locations': 2250,
        'transport_cost_range': (300, 1500),
        'max_center_capacity': 2100,
        'max_demand': 200,
        'max_urgency': 350,
        'operation_cost': 5000,
        'geo_radius': 0.7,
    }

    supply_allocation = MedicalSupplyAllocation(parameters, seed=seed)
    instance = supply_allocation.generate_instance()
    solve_status, solve_time = supply_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")