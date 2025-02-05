import random
import numpy as np
import time
from pyscipopt import Model, quicksum


class FruitDistributionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_suppliers > 0 and self.n_facilities > 0 and self.n_clients > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_processing_cost >= 0 and self.max_processing_cost >= self.min_processing_cost
        assert self.min_supplier_capacity > 0 and self.max_supplier_capacity >= self.min_supplier_capacity
        assert self.min_facility_capacity > 0 and self.max_facility_capacity >= self.min_facility_capacity
        assert self.min_client_demand > 0 and self.max_client_demand >= self.min_client_demand
        
        supplier_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_suppliers, self.n_facilities))
        facility_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_clients))
        processing_costs = np.random.randint(self.min_processing_cost, self.max_processing_cost + 1, self.n_facilities)
        supplier_capacities = np.random.randint(self.min_supplier_capacity, self.max_supplier_capacity + 1, self.n_suppliers)
        facility_capacities = np.random.randint(self.min_facility_capacity, self.max_facility_capacity + 1, self.n_facilities)
        client_demands = np.random.randint(self.min_client_demand, self.max_client_demand + 1, self.n_clients)
        
        return {
            "supplier_costs": supplier_costs,
            "facility_costs": facility_costs,
            "processing_costs": processing_costs,
            "supplier_capacities": supplier_capacities,
            "facility_capacities": facility_capacities,
            "client_demands": client_demands,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        supplier_costs = instance['supplier_costs']
        facility_costs = instance['facility_costs']
        processing_costs = instance['processing_costs']
        supplier_capacities = instance['supplier_capacities']
        facility_capacities = instance['facility_capacities']
        client_demands = instance['client_demands']
        
        model = Model("FruitDistributionOptimization")
        n_suppliers = len(supplier_costs)
        n_facilities = len(facility_costs)
        n_clients = len(client_demands)
        
        # Decision variables
        supplier_facility_vars = {(s, f): model.addVar(vtype="C", name=f"Supplier_{s}_Facility_{f}") for s in range(n_suppliers) for f in range(n_facilities)}
        facility_client_vars = {(f, c): model.addVar(vtype="C", name=f"Facility_{f}_Client_{c}") for f in range(n_facilities) for c in range(n_clients)}
        facility_active_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}_Active") for f in range(n_facilities)}
        overflow = model.addVar(vtype="C", lb=0, name="Overflow")
        
        # Objective: minimize the total cost (transport + processing costs + overflow penalties)
        model.setObjective(
            quicksum(supplier_costs[s, f] * supplier_facility_vars[s, f] for s in range(n_suppliers) for f in range(n_facilities)) +
            quicksum(facility_costs[f, c] * facility_client_vars[f, c] for f in range(n_facilities) for c in range(n_clients)) +
            quicksum(processing_costs[f] * facility_active_vars[f] for f in range(n_facilities)) +
            1000 * overflow, "minimize"
        )
        
        # Constraints: Supplier capacities
        for s in range(n_suppliers):
            model.addCons(quicksum(supplier_facility_vars[s, f] for f in range(n_facilities)) <= supplier_capacities[s], f"Supplier_{s}_Capacity")
        
        # Constraints: Facility capacities
        for f in range(n_facilities):
            model.addCons(
                quicksum(supplier_facility_vars[s, f] for s in range(n_suppliers)) +
                quicksum(facility_client_vars[f, c] for c in range(n_clients)) <= facility_capacities[f] + overflow, f"Facility_{f}_Capacity"
            )
        
        # Constraints: Client demands
        for c in range(n_clients):
            model.addCons(quicksum(facility_client_vars[f, c] for f in range(n_facilities)) >= client_demands[c], f"Client_{c}_Demand")
        
        # Constraints: Activate facility if any product is processed
        for f in range(n_facilities):
            model.addCons(
                quicksum(supplier_facility_vars[s, f] for s in range(n_suppliers)) + quicksum(facility_client_vars[f, c] for c in range(n_clients)) <= 
                facility_active_vars[f] * facility_capacities[f], f"Facility_{f}_Activation"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_suppliers': 400,
        'n_facilities': 200,
        'n_clients': 90,
        'min_transport_cost': 350,
        'max_transport_cost': 500,
        'min_processing_cost': 100,
        'max_processing_cost': 150,
        'min_supplier_capacity': 300,
        'max_supplier_capacity': 600,
        'min_facility_capacity': 300,
        'max_facility_capacity': 1000,
        'min_client_demand': 300,
        'max_client_demand': 900,
    }

    fruit_distribution_optimizer = FruitDistributionOptimization(parameters, seed=42)
    instance = fruit_distribution_optimizer.generate_instance()
    solve_status, solve_time, objective_value = fruit_distribution_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")