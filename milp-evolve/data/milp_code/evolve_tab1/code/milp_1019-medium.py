import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class WarehouseLayoutOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_units > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_facility_space > 0 and self.max_facility_space >= self.min_facility_space
        
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_units))
        spaces = np.random.randint(self.min_facility_space, self.max_facility_space + 1, self.n_facilities)
        demands = np.random.randint(1, 10, self.n_units)
        link_capacities = np.random.randint(self.min_link_capacity, self.max_link_capacity + 1, (self.n_facilities, self.n_units))
        premium_status = np.random.choice([0, 1], size=self.n_facilities, p=[0.8, 0.2])
        premium_transport_costs = np.random.randint(self.premium_transport_cost, self.premium_transport_cost + self.max_transport_cost + 1, (self.n_facilities, self.n_units))
        
        return {
            "facility_costs": facility_costs,
            "transport_costs": transport_costs,
            "spaces": spaces,
            "demands": demands,
            "link_capacities": link_capacities,
            "premium_status": premium_status,
            "premium_transport_costs": premium_transport_costs
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        transport_costs = instance['transport_costs']
        spaces = instance['spaces']
        demands = instance['demands']
        link_capacities = instance['link_capacities']
        premium_status = instance['premium_status']
        premium_transport_costs = instance["premium_transport_costs"]

        model = Model("WarehouseLayoutOptimization")
        n_facilities = len(facility_costs)
        n_units = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="C", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        premium_vars = {f: model.addVar(vtype="B", name=f"Premium_{f}") for f in range(n_facilities) if premium_status[f] == 1}

        # Objective: minimize the total cost (facility + transport)
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum((premium_transport_costs[f, u] if premium_status[f] == 1 else transport_costs[f, u]) * transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)), 
            "minimize"
        )
        
        # New constraint: Flow conservation
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) == demands[u], f"Unit_{u}_Demand")
        
        # New constraint: Link capacity
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= link_capacities[f, u], f"Link_{f}_{u}_Capacity")
        
        # Constraint: Only open facilities can serve units
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f] * demands[u], f"Facility_{f}_Serve_{u}")

        # Constraints: Facilities cannot exceed their space
        for f in range(n_facilities):
            model.addCons(quicksum(transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")

        # Big M constraints to introduce premium transportation costs conditionally
        M = max(self.max_transport_cost, self.premium_transport_cost)
        for f in range(n_facilities):
            if premium_status[f] == 1:
                model.addCons(premium_vars[f] <= facility_vars[f], f"Premium_indicate_{f}")
                for u in range(n_units):
                    model.addCons(transport_vars[f, u] <= M * premium_vars[f], f"Premium_transport_{f}_{u}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 54,
        'n_units': 840,
        'min_transport_cost': 1890,
        'max_transport_cost': 3000,
        'min_facility_cost': 2500,
        'max_facility_cost': 5000,
        'min_facility_space': 660,
        'max_facility_space': 2400,
        'min_link_capacity': 280,
        'max_link_capacity': 1890,
        'premium_transport_cost': 5000,
    }

    warehouse_layout_optimizer = WarehouseLayoutOptimization(parameters, seed=42)
    instance = warehouse_layout_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_layout_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")