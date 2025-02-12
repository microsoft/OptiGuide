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
        demands = np.random.normal(self.avg_demand, self.demand_stddev, self.n_units).astype(int)
        
        # Additional data for realistic complexity
        compatibility = np.random.randint(0, 2, (self.n_facilities, self.n_units))
        importance = np.random.rand(self.n_units)
        
        return {
            "facility_costs": facility_costs,
            "transport_costs": transport_costs,
            "spaces": spaces,
            "demands": demands,
            "compatibility": compatibility,
            "importance": importance,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        transport_costs = instance['transport_costs']
        spaces = instance['spaces']
        demands = instance['demands']
        compatibility = instance['compatibility']
        importance = instance['importance']
        
        model = Model("WarehouseLayoutOptimization")
        n_facilities = len(facility_costs)
        n_units = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        aux_vars = {(f, u): model.addVar(vtype="B", name=f"Aux_Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        
        # Objective: minimize the total cost (facility + transport) weighted by importance
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f, u] * transport_vars[f, u] * importance[u] for f in range(n_facilities) for u in range(n_units)), 
            "minimize"
        )
        
        # Constraints: Each unit demand is met rigorously using set partitioning
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) == 1, f"Unit_{u}_Demand")
        
        # Convex Hull Constraints: Only open facilities can serve units with auxiliary variables for tighter formulation
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f], f"Facility_{f}_Serve_{u}")
                model.addCons(aux_vars[f, u] == transport_vars[f, u] * facility_vars[f], f"Aux_Constraint_{f}_{u}")

        # Constraints: Facilities cannot exceed their space
        for f in range(n_facilities):
            model.addCons(quicksum(demands[u] * transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")
        
        # Additional constraints: Ensuring compatibility between facility and unit
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= compatibility[f, u], f"Compatibility_Constraint_{f}_{u}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 900,
        'n_units': 42,
        'min_transport_cost': 30,
        'max_transport_cost': 112,
        'min_facility_cost': 1500,
        'max_facility_cost': 5000,
        'min_facility_space': 8,
        'max_facility_space': 40,
        'avg_demand': 6,
        'demand_stddev': 0,
    }

    warehouse_layout_optimizer = WarehouseLayoutOptimization(parameters, seed=42)
    instance = warehouse_layout_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_layout_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")