import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocationWithInventoryManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_demand_points > 0
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_inventory_cost >= 0 and self.max_inventory_cost >= self.min_inventory_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_demand > 0 and self.max_demand >= self.min_demand

        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        inventory_costs = np.random.randint(self.min_inventory_cost, self.max_inventory_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_demand_points))

        demands = np.random.randint(self.min_demand, self.max_demand + 1, self.n_demand_points)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)

        return {
            "facility_costs": facility_costs,
            "inventory_costs": inventory_costs,
            "transport_costs": transport_costs,
            "demands": demands,
            "capacities": capacities,
        }

    def solve(self, instance):
        facility_costs = instance['facility_costs']
        inventory_costs = instance['inventory_costs']
        transport_costs = instance['transport_costs']
        demands = instance['demands']
        capacities = instance['capacities']

        model = Model("FacilityLocationWithInventoryManagement")
        n_facilities = len(facility_costs)
        n_demand_points = len(demands)
        
        # Variables
        open_vars = {f: model.addVar(vtype="B", name=f"FacilityOpen_{f}") for f in range(n_facilities)}
        inventory_vars = {f: model.addVar(vtype="C", name=f"Inventory_{f}", lb=0) for f in range(n_facilities)}
        transport_vars = {(f, d): model.addVar(vtype="B", name=f"Transport_{f}_{d}") for f in range(n_facilities) for d in range(n_demand_points)}
        
        # Objective Function
        model.setObjective(
            quicksum(facility_costs[f] * open_vars[f] for f in range(n_facilities)) +
            quicksum(inventory_costs[f] * inventory_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f][d] * transport_vars[f, d] for f in range(n_facilities) for d in range(n_demand_points)),
            "minimize"
        )

        # Constraints
        for d in range(n_demand_points):
            model.addCons(quicksum(transport_vars[f, d] for f in range(n_facilities)) == 1, f"Demand_{d}")

        for f in range(n_facilities):
            model.addCons(quicksum(demands[d] * transport_vars[f, d] for d in range(n_demand_points)) <= capacities[f] * open_vars[f], f"Capacity_{f}")
        
        for f in range(n_facilities):
            model.addCons(inventory_vars[f] <= capacities[f] * open_vars[f], f"InventoryCapacity_{f}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    # New problem parameters
    seed = 42
    parameters = {
        'n_facilities': 50,
        'n_demand_points': 200,
        'min_facility_cost': 500,
        'max_facility_cost': 1000,
        'min_inventory_cost': 10,
        'max_inventory_cost': 50,
        'min_transport_cost': 1,
        'max_transport_cost': 10,
        'min_demand': 20,
        'max_demand': 100,
        'min_capacity': 1000,
        'max_capacity': 5000,
    }

    facility_optimizer = FacilityLocationWithInventoryManagement(parameters, seed=42)
    instance = facility_optimizer.generate_instance()
    solve_status, solve_time, objective_value = facility_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")