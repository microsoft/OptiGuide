import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HospitalBedAllocation:
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
        hospital_capacities = [random.randint(1, self.max_hospital_capacity) for _ in range(self.n_hospitals)]
        patient_demand = [random.randint(1, self.max_demand) for _ in range(self.n_patients)]

        for i, j in G.edges:
            cost = np.random.uniform(*self.service_cost_range)
            adj_mat[i, j] = cost
            edge_list.append((i, j))

        hospitals = range(self.n_hospitals)
        patients = range(self.n_patients, self.n_total_nodes)
        
        return G, adj_mat, edge_list, hospital_capacities, patient_demand, hospitals, patients

    def generate_instance(self):
        self.n_total_nodes = self.n_hospitals + self.n_patients
        G, adj_mat, edge_list, hospital_capacities, patient_demand, hospitals, patients = self.generate_city_graph()

        res = {
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'hospital_capacities': hospital_capacities,
            'patient_demand': patient_demand,
            'hospitals': hospitals,
            'patients': patients
        }
        # New instance data
        z_weights = {f"Convex_Comb_{i}": np.random.uniform(0, 1) for i in range(self.n_combination_vars)}
        res['z_weights'] = z_weights

        return res
    
    # PySCIPOpt Modeling
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        hospital_capacities = instance['hospital_capacities']
        patient_demand = instance['patient_demand']
        hospitals = instance['hospitals']
        patients = instance['patients']
        z_weights = instance['z_weights']
        
        model = Model("HospitalBedAllocation")
        y_vars = {f"New_Hospital_{h+1}": model.addVar(vtype="B", name=f"New_Hospital_{h+1}") for h in hospitals}
        x_vars = {f"Hospital_Assign_{h+1}_{p+1}": model.addVar(vtype="B", name=f"Hospital_Assign_{h+1}_{p+1}") for h in hospitals for p in patients}
        z_vars = {f"Convex_Comb_{i}": model.addVar(vtype="C", name=f"Convex_Comb_{i}") for i in range(self.n_combination_vars)}
        
        # Objective function: minimize the total operation and service cost
        objective_expr = quicksum(
            adj_mat[h, p] * x_vars[f"Hospital_Assign_{h+1}_{p+1}"]
            for h in hospitals for p in patients
        )
        # Adding maintenance cost for opening a hospital
        objective_expr += quicksum(
            self.operation_cost * y_vars[f"New_Hospital_{h+1}"]
            for h in hospitals
        )
        model.setObjective(objective_expr, "minimize")

        # Constraints
        # Each patient is assigned to at least one hospital (Set Covering constraint)
        for p in patients:
            model.addCons(quicksum(x_vars[f"Hospital_Assign_{h+1}_{p+1}"] for h in hospitals) >= 1, f"Serve_{p+1}")

        # Hospital should be open if it serves any patient
        for h in hospitals:
            for p in patients:
                model.addCons(x_vars[f"Hospital_Assign_{h+1}_{p+1}"] <= y_vars[f"New_Hospital_{h+1}"], f"Open_Cond_{h+1}_{p+1}")

        # Hospital capacity constraint
        for h in hospitals:
            model.addCons(quicksum(patient_demand[p-self.n_patients] * x_vars[f"Hospital_Assign_{h+1}_{p+1}"] for p in patients) <= hospital_capacities[h], f"Capacity_{h+1}")

        # Convex hull constraints to add complexity
        for i in range(self.n_combination_vars):
            model.addCons(quicksum(z_weights[f"Convex_Comb_{j}"] * hospital_capacities[j] 
                                   for j in range(self.n_combination_vars)) >= quicksum(patient_demand[p-self.n_patients] * x_vars[f"Hospital_Assign_{i+1}_{p+1}"] 
                                   for p in patients for i in range(self.n_combination_vars)), f"Convex_Combination_{i+1}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hospitals': 50,
        'n_patients': 500,
        'service_cost_range': (150, 750),
        'max_hospital_capacity': 375,
        'max_demand': 150,
        'operation_cost': 7000,
        'geo_radius': 0.38,
        'n_combination_vars': 10
    }

    hospital_allocation = HospitalBedAllocation(parameters, seed=seed)
    instance = hospital_allocation.generate_instance()
    solve_status, solve_time = hospital_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")