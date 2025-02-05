import random
import time
import numpy as np
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

class HealthcareLogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_distribution_centers(self):
        n_nodes = np.random.randint(self.min_centers, self.max_centers)
        centers = np.random.choice(range(1000), size=n_nodes, replace=False)  # Random unique center IDs
        return centers

    def generate_hospitals(self, n_centers):
        n_hospitals = np.random.randint(self.min_hospitals, self.max_hospitals)
        hospitals = np.random.choice(range(1000, 2000), size=n_hospitals, replace=False)
        demands = {node: np.random.randint(self.min_demand, self.max_demand) for node in hospitals}
        return hospitals, demands

    def generate_transportation_costs(self, centers, hospitals):
        costs = {(c, h): np.random.uniform(self.min_transport_cost, self.max_transport_cost) for c in centers for h in hospitals}
        return costs

    def generate_storage_capacities(self, centers):
        capacities = {c: np.random.randint(self.min_capacity, self.max_capacity) for c in centers}
        return capacities

    def get_instance(self):
        centers = self.generate_distribution_centers()
        hospitals, demands = self.generate_hospitals(len(centers))
        transport_costs = self.generate_transportation_costs(centers, hospitals)
        capacities = self.generate_storage_capacities(centers)
        
        center_costs = {c: np.random.uniform(self.min_center_cost, self.max_center_cost) for c in centers}
        storage_costs = {c: np.random.uniform(self.min_storage_cost, self.max_storage_cost) for c in centers}
        emergency_costs = {c: np.random.uniform(self.min_emergency_cost, self.max_emergency_cost) for c in centers}
        
        return {
            'centers': centers,
            'hospitals': hospitals,
            'transport_costs': transport_costs,
            'capacities': capacities,
            'demands': demands,
            'center_costs': center_costs,
            'storage_costs': storage_costs,
            'emergency_costs': emergency_costs,
        }

    def solve(self, instance):
        centers = instance['centers']
        hospitals = instance['hospitals']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        center_costs = instance['center_costs']
        storage_costs = instance['storage_costs']
        emergency_costs = instance['emergency_costs']

        model = Model("HealthcareLogisticsOptimization")

        Nodes_location_vars = {c: model.addVar(vtype="B", name=f"Node_Loc_{c}") for c in centers}
        NewAllocation_vars = {(c, h): model.addVar(vtype="B", name=f"NewAlloc_{c}_{h}") for c in centers for h in hospitals}
        Transportation_vars = {(c, h): model.addVar(vtype="C", name=f"Trans_{c}_{h}") for c in centers for h in hospitals}
        Hospital_vars = {c: model.addVar(vtype="C", name=f"Hospital_{c}") for c in centers}
        TemporaryStorage_vars = {c: model.addVar(vtype="C", name=f"TempStorage_{c}") for c in centers}

        # Objective function
        total_cost = quicksum(Nodes_location_vars[c] * center_costs[c] for c in centers)
        total_cost += quicksum(NewAllocation_vars[c, h] * transport_costs[c, h] for c in centers for h in hospitals)
        total_cost += quicksum(Hospital_vars[c] * storage_costs[c] for c in centers)
        total_cost += quicksum(TemporaryStorage_vars[c] * emergency_costs[c] for c in centers)
        
        model.setObjective(total_cost, "minimize")

        # Constraints
        for c in centers:
            model.addCons(
                quicksum(Transportation_vars[c, h] for h in hospitals) <= capacities[c] * Nodes_location_vars[c],
                name=f"Capacity_{c}"
            )

        for h in hospitals:
            model.addCons(
                quicksum(Transportation_vars[c, h] for c in centers) >= demands[h],
                name=f"HospitalDemand_{h}"
            )

        for c in centers:
            model.addCons(
                Hospital_vars[c] <= capacities[c],
                name=f"StorageLimit_{c}"
            )

        for c in centers:
            model.addCons(
                TemporaryStorage_vars[c] <= capacities[c] * 0.15,  # Assume temporary storage is limited to 15% of the capacity
                name=f"TempStorageLimit_{c}"
            )

        model.setParam('limits/time', 10 * 60)  # Set a time limit of 10 minutes for solving
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_centers': 75,
        'max_centers': 1000,
        'min_hospitals': 25,
        'max_hospitals': 600,
        'min_demand': 300,
        'max_demand': 800,
        'min_transport_cost': 0.5,
        'max_transport_cost': 360.0,
        'min_capacity': 225,
        'max_capacity': 1800,
        'min_center_cost': 30000,
        'max_center_cost': 150000,
        'min_storage_cost': 2000,
        'max_storage_cost': 4000,
        'min_emergency_cost': 250,
        'max_emergency_cost': 2000,
    }
    optimizer = HealthcareLogisticsOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")