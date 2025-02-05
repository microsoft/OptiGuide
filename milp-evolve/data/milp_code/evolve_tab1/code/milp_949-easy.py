import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimplifiedSupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_demand_points > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_facility_capacity > 0 and self.max_facility_capacity >= self.min_facility_capacity
        
        facility_opening_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_demand_points))
        capacities = np.random.randint(self.min_facility_capacity, self.max_facility_capacity + 1, self.n_facilities)
        demand = np.random.randint(1, 50, self.n_demand_points)
        
        return {
            "facility_opening_costs": facility_opening_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "demand": demand,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_opening_costs = instance['facility_opening_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demand = instance['demand']
        
        model = Model("SimplifiedSupplyChainOptimization")
        n_facilities = len(facility_opening_costs)
        n_demand_points = len(demand)
        
        # Decision variables
        facility_open_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        assignment_vars = {(f, d): model.addVar(vtype="B", name=f"Facility_{f}_Demand_{d}") for f in range(n_facilities) for d in range(n_demand_points)}
        
        # Objective: minimize the total cost (facility opening + transport costs)
        model.setObjective(
            quicksum(facility_opening_costs[f] * facility_open_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f, d] * assignment_vars[f, d] for f in range(n_facilities) for d in range(n_demand_points)),
            "minimize"
        )
        
        # Constraints: Each demand point is assigned to exactly one facility
        for d in range(n_demand_points):
            model.addCons(quicksum(assignment_vars[f, d] for f in range(n_facilities)) == 1, f"Demand_{d}_Assignment")
        
        # Constraints: Facility capacity constraints
        for f in range(n_facilities):
            model.addCons(quicksum(demand[d] * assignment_vars[f, d] for d in range(n_demand_points)) <= capacities[f] * facility_open_vars[f], f"Facility_{f}_Capacity")
        
        # Constraints: Facilities must be open to accept assignments
        for f in range(n_facilities):
            for d in range(n_demand_points):
                model.addCons(assignment_vars[f, d] <= facility_open_vars[f], f"Open_Facility_{f}_For_Demand_{d}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 180,
        'n_demand_points': 60,
        'min_transport_cost': 15,
        'max_transport_cost': 500,
        'min_facility_cost': 3000,
        'max_facility_cost': 5000,
        'min_facility_capacity': 100,
        'max_facility_capacity': 5000,
    }

    supply_chain_optimizer = SimplifiedSupplyChainOptimization(parameters, seed=42)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")