import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class UrbanHousingManagement:
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
        assert self.min_management_cost >= 0 and self.max_management_cost >= self.min_management_cost
        assert self.min_infrastructure_cost >= 0 and self.max_infrastructure_cost >= self.min_infrastructure_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        
        management_costs = np.random.randint(self.min_management_cost, self.max_management_cost + 1, self.n_facilities)
        infrastructure_costs = np.random.randint(self.min_infrastructure_cost, self.max_infrastructure_cost + 1, (self.n_facilities, self.n_units))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)
        quality_weights = np.random.uniform(0.8, 1.0, (self.n_facilities, self.n_units))
        surge_penalties = np.random.randint(100, 500, self.n_units)
        
        return {
            "management_costs": management_costs,
            "infrastructure_costs": infrastructure_costs,
            "capacities": capacities,
            "quality_weights": quality_weights,
            "surge_penalties": surge_penalties
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        management_costs = instance['management_costs']
        infrastructure_costs = instance['infrastructure_costs']
        capacities = instance['capacities']
        quality_weights = instance['quality_weights']
        surge_penalties = instance['surge_penalties']
        
        model = Model("UrbanHousingManagement")
        n_facilities = len(management_costs)
        n_units = len(infrastructure_costs[0])
        big_m = max(infrastructure_costs.flatten())
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        unit_connection_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        unit_backup_vars = {(f, u): model.addVar(vtype="B", name=f"Backup_Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        urban_surge = model.addVar(vtype="C", lb=0, name="UrbanSurge")
        
        # Objective: minimize the total costs (management + infrastructure + urban surge + backup penalties)
        model.setObjective(quicksum(management_costs[f] * facility_vars[f] for f in range(n_facilities)) +
                           quicksum(infrastructure_costs[f, u] * unit_connection_vars[f, u] for f in range(n_facilities) for u in range(n_units)) +
                           1000 * urban_surge +
                           quicksum(surge_penalties[u] * unit_backup_vars[f, u] for f in range(n_facilities) for u in range(n_units)), "minimize")
        
        # Constraints: Each housing unit is connected to exactly one facility
        for u in range(n_units):
            model.addCons(quicksum(unit_connection_vars[f, u] for f in range(n_facilities)) == 1, f"Unit_{u}_Assignment")
        
        # Constraints: Only established facilities can connect to units
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(unit_connection_vars[f, u] <= facility_vars[f], f"Facility_{f}_Connection_{u}")
        
        # Constraints: Each unit can only have one backup facility
        for u in range(n_units):
            model.addCons(quicksum(unit_backup_vars[f, u] for f in range(n_facilities)) <= 1, f"Unit_{u}_Backup")
        
        # Constraints: Facilities cannot exceed their capacity
        for f in range(n_facilities):
            model.addCons(quicksum(unit_connection_vars[f, u] for u in range(n_units)) <= capacities[f] + urban_surge, f"Facility_{f}_Capacity")
        
        # New Constraint: Housing units must have a minimum quality of living
        min_quality = 0.9
        for u in range(n_units):
            model.addCons(quicksum(quality_weights[f, u] * unit_connection_vars[f, u] for f in range(n_facilities)) >= min_quality, f"Unit_{u}_Quality")
        
        # New Constraint: Facilities must serve at least a minimum number of housing units
        min_service = 5
        for f in range(n_facilities):
            model.addCons(quicksum(unit_connection_vars[f, u] for u in range(n_units)) >= facility_vars[f] * min_service, f"Facility_{f}_MinService")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 81,
        'n_units': 100,
        'min_management_cost': 3000,
        'max_management_cost': 10000,
        'min_infrastructure_cost': 900,
        'max_infrastructure_cost': 937,
        'min_capacity': 1008,
        'max_capacity': 3000,
    }

    urban_optimizer = UrbanHousingManagement(parameters, seed=42)
    instance = urban_optimizer.generate_instance()
    solve_status, solve_time, objective_value = urban_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")