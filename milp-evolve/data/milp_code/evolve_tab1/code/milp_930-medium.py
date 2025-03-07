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
        opening_times = np.random.randint(self.min_opening_time, self.max_opening_time + 1, self.n_facilities)
        maintenance_periods = np.random.randint(self.min_maintenance_period, self.max_maintenance_period + 1, self.n_facilities)
        
        return {
            "facility_costs": facility_costs,
            "transport_costs": transport_costs,
            "spaces": spaces,
            "demands": demands,
            "opening_times": opening_times,
            "maintenance_periods": maintenance_periods,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        transport_costs = instance['transport_costs']
        spaces = instance['spaces']
        demands = instance['demands']
        opening_times = instance['opening_times']
        maintenance_periods = instance['maintenance_periods']
        
        model = Model("WarehouseLayoutOptimization")
        n_facilities = len(facility_costs)
        n_units = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        opening_time_vars = {f: model.addVar(vtype="I", lb=0, name=f"OpeningTime_{f}") for f in range(n_facilities)}
        maintenance_period_vars = {f: model.addVar(vtype="I", lb=0, name=f"MaintenancePeriod_{f}") for f in range(n_facilities)}
        
        # Objective: minimize the total cost and unused space penalties, while maximizing load balance
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f, u] * transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)) +
            quicksum(opening_times[f] * opening_time_vars[f] for f in range(n_facilities)) +
            quicksum(maintenance_periods[f] * maintenance_period_vars[f] for f in range(n_facilities)) -
            50 * quicksum(transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)) 
            , "minimize"
        )
        
        # Constraints: Each unit demand is met by exactly one facility
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) == 1, f"Unit_{u}_Demand")
        
        # Constraints: Only open facilities can serve units
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f], f"Facility_{f}_Serve_{u}")
        
        # Constraints: Facilities cannot exceed their space
        for f in range(n_facilities):
            model.addCons(quicksum(demands[u] * transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")
        
        # Constraints: Facilities must respect their opening and maintenance schedules
        for f in range(n_facilities):
            model.addCons(opening_time_vars[f] == opening_times[f], f"Facility_{f}_OpeningTime")
            model.addCons(maintenance_period_vars[f] == maintenance_periods[f], f"Facility_{f}_MaintenancePeriod")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 100,
        'n_units': 112,
        'min_transport_cost': 2,
        'max_transport_cost': 100,
        'min_facility_cost': 1500,
        'max_facility_cost': 5000,
        'min_facility_space': 2250,
        'max_facility_space': 2400,
        'min_opening_time': 1,
        'max_opening_time': 2,
        'min_maintenance_period': 90,
        'max_maintenance_period': 180,
    }
    
    warehouse_layout_optimizer = WarehouseLayoutOptimization(parameters, seed=42)
    instance = warehouse_layout_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_layout_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")