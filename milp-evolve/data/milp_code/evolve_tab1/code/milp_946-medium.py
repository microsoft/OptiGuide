import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MedicalCenterAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_medical_centers > 0 and self.n_neighborhoods > 0
        assert self.min_hiring_cost >= 0 and self.max_hiring_cost >= self.min_hiring_cost
        assert self.min_oper_supply >= 0 and self.max_oper_supply >= self.min_oper_supply
        assert self.min_quality_assurance_cost > 0 and self.max_quality_assurance_cost >= self.min_quality_assurance_cost
        
        hiring_costs = np.random.randint(self.min_hiring_cost, self.max_hiring_cost + 1, self.n_medical_centers)
        operational_supplies = np.random.randint(self.min_oper_supply, self.max_oper_supply + 1, (self.n_medical_centers, self.n_neighborhoods))
        capacities = np.random.randint(self.min_quality_assurance_cost, self.max_quality_assurance_cost + 1, self.n_medical_centers)
        demands = np.random.randint(1, 10, self.n_neighborhoods)
        
        return {
            "hiring_costs": hiring_costs,
            "operational_supplies": operational_supplies,
            "capacities": capacities,
            "demands": demands
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        hiring_costs = instance['hiring_costs']
        operational_supplies = instance['operational_supplies']
        capacities = instance['capacities']
        demands = instance['demands']
        
        model = Model("MedicalCenterAllocation")
        n_med_centers = len(hiring_costs)
        n_neighborhoods = len(operational_supplies[0])
        
        # Decision variables
        medical_center_vars = {m: model.addVar(vtype="B", name=f"MedicalCenter_{m}") for m in range(n_med_centers)}
        serve_vars = {(m, n): model.addVar(vtype="B", name=f"MedicalCenter_{m}_Neighborhood_{n}") for m in range(n_med_centers) for n in range(n_neighborhoods)}
        
        # Objective: Minimize the total costs while ensuring quality service
        ### new constraints and variables and objective code additions start here
        model.setObjective(
            quicksum(hiring_costs[m] * medical_center_vars[m] for m in range(n_med_centers)) +
            quicksum(operational_supplies[m, n] * serve_vars[m, n] for m in range(n_med_centers) for n in range(n_neighborhoods)) +
            quicksum(100 * serve_vars[m, n] for m in range(n_med_centers) for n in range(n_neighborhoods)), "minimize"
        )
        
        # Constraints: Each neighborhood's healthcare demand is met by exactly one medical center
        for n in range(n_neighborhoods):
            model.addCons(quicksum(serve_vars[m, n] for m in range(n_med_centers)) == 1, f"Neighborhood_{n}_Demand")
        
        # Constraints: Only open medical centers can serve neighborhoods
        for m in range(n_med_centers):
            for n in range(n_neighborhoods):
                model.addCons(serve_vars[m, n] <= medical_center_vars[m], f"MedicalCenter_{m}_Serve_{n}")
        
        # Constraints: Medical centers cannot exceed their capacity
        for m in range(n_med_centers):
            model.addCons(quicksum(demands[n] * serve_vars[m, n] for n in range(n_neighborhoods)) <= capacities[m], f"MedicalCenter_{m}_Capacity")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_medical_centers': 60,
        'n_neighborhoods': 250,
        'min_hiring_cost': 500,
        'max_hiring_cost': 3000,
        'min_oper_supply': 50,
        'max_oper_supply': 5000,
        'min_quality_assurance_cost': 500,
        'max_quality_assurance_cost': 8000,
    }

    medical_center_allocator = MedicalCenterAllocation(parameters, seed=42)
    instance = medical_center_allocator.generate_instance()
    solve_status, solve_time, objective_value = medical_center_allocator.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")