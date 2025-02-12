import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HealthcareNetworkDesignOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_clinics = random.randint(self.min_clinics, self.max_clinics)
        num_units = random.randint(self.min_units, self.max_units)

        transport_cost = np.random.randint(50, 300, size=(num_units, num_clinics))
        opening_costs_clinic = np.random.randint(1000, 3000, size=num_clinics)
        maintenance_costs_clinic = np.random.randint(200, 500, size=num_clinics)
        
        patient_demands = np.random.randint(100, 500, size=num_units)
        clinic_capacity = np.random.randint(5000, 10000, size=num_clinics)
        
        max_acceptable_distance = np.random.randint(50, 200)
        unit_distances_from_clinics = np.random.randint(10, 500, size=(num_units, num_clinics))

        customer_service_hours = np.random.randint(5, 15, size=(num_units, num_clinics))

        res = {
            'num_clinics': num_clinics,
            'num_units': num_units,
            'transport_cost': transport_cost,
            'opening_costs_clinic': opening_costs_clinic,
            'maintenance_costs_clinic': maintenance_costs_clinic,
            'patient_demands': patient_demands,
            'clinic_capacity': clinic_capacity,
            'unit_distances_from_clinics': unit_distances_from_clinics,
            'max_acceptable_distance': max_acceptable_distance,
            'customer_service_hours': customer_service_hours,
        }

        G = nx.erdos_renyi_graph(num_units + num_clinics, self.highway_connectivity_prob, seed=self.seed)
        res['highway_network'] = list(G.edges)

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_clinics = instance['num_clinics']
        num_units = instance['num_units']
        transport_cost = instance['transport_cost']
        opening_costs_clinic = instance['opening_costs_clinic']
        maintenance_costs_clinic = instance['maintenance_costs_clinic']
        patient_demands = instance['patient_demands']
        clinic_capacity = instance['clinic_capacity']
        unit_distances_from_clinics = instance['unit_distances_from_clinics']
        max_acceptable_distance = instance['max_acceptable_distance']
        customer_service_hours = instance['customer_service_hours']
        highway_network = instance['highway_network']

        model = Model("HealthcareNetworkDesignOptimization")

        # Variables
        Clinics_Opened = {j: model.addVar(vtype="B", name=f"Clinics_Opened_{j}") for j in range(num_clinics)}
        Patient_Assignment = {(i, j): model.addVar(vtype="B", name=f"Patient_Assignment_{i}_{j}") for i in range(num_units) for j in range(num_clinics)}
        Highways_Open = {(u, v): model.addVar(vtype="B", name=f"Highways_Open_{u}_{v}") for u, v in highway_network}

        # Objective function: Minimize total costs
        total_cost = quicksum(Patient_Assignment[i, j] * transport_cost[i, j] for i in range(num_units) for j in range(num_clinics)) + \
                     quicksum(Clinics_Opened[j] * opening_costs_clinic[j] for j in range(num_clinics)) + \
                     quicksum(Clinics_Opened[j] * maintenance_costs_clinic[j] for j in range(num_clinics))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_units):
            model.addCons(quicksum(Patient_Assignment[i, j] for j in range(num_clinics)) == 1, name=f"unit_served_{i}")

        for j in range(num_clinics):
            for i in range(num_units):
                model.addCons(Patient_Assignment[i, j] <= Clinics_Opened[j], name=f"clinic_open_{i}_{j}")

        for j in range(num_clinics):
            model.addCons(quicksum(patient_demands[i] * Patient_Assignment[i, j] for i in range(num_units)) <= clinic_capacity[j], name=f"clinic_capacity_{j}")
        
        for i in range(num_units):
            for j in range(num_clinics):
                model.addCons(unit_distances_from_clinics[i, j] * Patient_Assignment[i, j] <= max_acceptable_distance, name=f"distance_constraint_{i}_{j}")

        min_service_hours = 5  # Example minimum service hours; can be a parameter
        for j in range(num_clinics):
            model.addCons(quicksum(customer_service_hours[i, j] * Patient_Assignment[i, j] for i in range(num_units)) >= min_service_hours * Clinics_Opened[j], name=f"min_service_hours_{j}")

        # Incorporate highway network constraints
        for u, v in highway_network:
            unit_clinic_assignment_u = quicksum(Patient_Assignment[u, j] for j in range(num_clinics) if u < num_units)
            unit_clinic_assignment_v = quicksum(Patient_Assignment[v, j] for j in range(num_clinics) if v < num_units)
            model.addCons(unit_clinic_assignment_u + unit_clinic_assignment_v <= Highways_Open[u, v] + 1, name=f"highway_network_{u}_{v}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_clinics': 37,
        'max_clinics': 75,
        'min_units': 100,
        'max_units': 200,
        'highway_connectivity_prob': 0.17,
    }
    optimization = HealthcareNetworkDesignOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")