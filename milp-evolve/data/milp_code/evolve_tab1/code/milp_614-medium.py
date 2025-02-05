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

        high_priority_nodes = random.sample(locations, len(locations) // 3)
        facility_count = self.n_facilities
        facility_assignment = np.random.randint(0, 2, (facility_count, len(locations))).tolist()
        patrol_fees = np.random.randint(50, 500, size=facility_count).tolist()

        res = {
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'distribution_center_capacities': distribution_center_capacities,
            'location_demand': location_demand,
            'distribution_centers': distribution_centers,
            'locations': locations,
            'facility_count': facility_count,
            'facility_assignment': facility_assignment,
            'patrol_fees': patrol_fees,
            'high_priority_nodes': high_priority_nodes
        }
        return res
    
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        distribution_center_capacities = instance['distribution_center_capacities']
        location_demand = instance['location_demand']
        distribution_centers = instance['distribution_centers']
        locations = instance['locations']
        facility_count = instance['facility_count']
        facility_assignment = instance['facility_assignment']
        patrol_fees = instance['patrol_fees']
        high_priority_nodes = instance['high_priority_nodes']
        
        model = Model("MedicalSupplyAllocation")
        y_vars = {f"New_Center_{c+1}": model.addVar(vtype="B", name=f"New_Center_{c+1}") for c in distribution_centers}
        x_vars = {f"Center_Assign_{c+1}_{l+1}": model.addVar(vtype="B", name=f"Center_Assign_{c+1}_{l+1}") for c in distribution_centers for l in locations}

        # Additional flow variables
        flow_vars = {(c, l): model.addVar(vtype="C", name=f"Flow_{c+1}_{l+1}") for c in distribution_centers for l in locations}

        # Facility assignment and patrol variables
        f_vars = {f"Facility_{f+1}": model.addVar(vtype="B", name=f"Facility_{f+1}") for f in range(facility_count)}
        p_vars = {f"Patrol_{f+1}_{l+1}": model.addVar(vtype="B", name=f"Patrol_{f+1}_{l+1}") for f in range(facility_count) for l in high_priority_nodes}
        
        # Adjusted objective function with flow, facility, and patrol costs
        objective_expr = quicksum(
            adj_mat[c, l] * x_vars[f"Center_Assign_{c+1}_{l+1}"]
            for c in distribution_centers for l in locations if adj_mat[c, l] != 0
        )
        # Adding maintenance cost for opening a distribution center
        objective_expr += quicksum(
            self.operation_cost * y_vars[f"New_Center_{c+1}"]
            for c in distribution_centers
        )
        # Adding flow and patrol costs
        objective_expr += quicksum(
            flow_vars[(c, l)] * adj_mat[c, l] for c in distribution_centers for l in locations if adj_mat[c, l] != 0
        )
        # Adding facility and patrol costs
        objective_expr += quicksum(
            patrol_fees[f] * p_vars[f"Patrol_{f+1}_{l+1}"] for f in range(facility_count) for l in high_priority_nodes
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
        
        # Flow conservation constraints
        for c in distribution_centers:
            model.addCons(quicksum(flow_vars[(c, l)] for l in locations) == quicksum(x_vars[f"Center_Assign_{c+1}_{l+1}"] for l in locations), f"FlowConservation_C_{c+1}")

        # Facility assignment for centers, ensuring feasibility and capacity constraints
        for f in range(facility_count):
            for l in high_priority_nodes:
                model.addCons(p_vars[f"Patrol_{f+1}_{l+1}"] <= f_vars[f"Facility_{f+1}"], f"PatrolAssignment_{f+1}_{l+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_centers': 60,
        'n_locations': 562,
        'transport_cost_range': (150, 750),
        'max_center_capacity': 1575,
        'max_demand': 20,
        'max_urgency': 175,
        'operation_cost': 5000,
        'geo_radius': 0.31,
        'n_facilities': 2,
    }

    supply_allocation = MedicalSupplyAllocation(parameters, seed=seed)
    instance = supply_allocation.generate_instance()
    solve_status, solve_time = supply_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")