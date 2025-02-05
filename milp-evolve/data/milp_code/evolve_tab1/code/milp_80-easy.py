import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityPlanningCostObjective:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_facilities(self):
        facilities = [{} for _ in range(self.n_facilities)]
        for i in range(self.n_facilities):
            facilities[i]['fixed_cost'] = np.random.uniform(self.facility_fixed_cost_min, self.facility_fixed_cost_max)
            facilities[i]['capacity'] = np.random.uniform(self.facility_capacity_min, self.facility_capacity_max)
        return facilities
    
    def generate_customers(self):
        customers = [{} for _ in range(self.n_customers)]
        for j in range(self.n_customers):
            customers[j]['demand'] = np.random.uniform(self.customer_demand_min, self.customer_demand_max)
        return customers
    
    def generate_transportation_costs(self):
        transport_costs = np.random.uniform(self.transport_cost_min, self.transport_cost_max, (self.n_facilities, self.n_customers))
        return transport_costs
    
    def generate_inventory_holding_costs(self):
        inventory_costs = np.random.uniform(self.inventory_cost_min, self.inventory_cost_max, self.n_facilities)
        return inventory_costs
    
    def generate_instance(self):
        self.n_facilities = np.random.randint(self.min_n_facilities, self.max_n_facilities + 1)
        self.n_customers = np.random.randint(self.min_n_customers, self.max_n_customers + 1)
        
        facilities = self.generate_facilities()
        customers = self.generate_customers()
        transport_costs = self.generate_transportation_costs()
        inventory_costs = self.generate_inventory_holding_costs()
        
        res = {
            'facilities': facilities,
            'customers': customers,
            'transport_costs': transport_costs,
            'inventory_costs': inventory_costs
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        facilities = instance['facilities']
        customers = instance['customers']
        transport_costs = instance['transport_costs']
        inventory_costs = instance['inventory_costs']
        
        model = Model("FacilityPlanningCostObjective")
        
        y_vars = {f"y_{i+1}": model.addVar(vtype="B", name=f"y_{i+1}") for i in range(self.n_facilities)}
        x_vars = {
            f"x_{i+1}_{j+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}")
            for i in range(self.n_facilities)
            for j in range(self.n_customers)
        }
        inventory_vars = {
            f"inventory_{i+1}": model.addVar(vtype="C", name=f"inventory_{i+1}")
            for i in range(self.n_facilities)
        }

        # Objective function
        fixed_cost_expr = quicksum(facilities[i]['fixed_cost'] * y_vars[f"y_{i+1}"] for i in range(self.n_facilities))
        transport_cost_expr = quicksum(transport_costs[i][j] * x_vars[f"x_{i+1}_{j+1}"] for i in range(self.n_facilities) for j in range(self.n_customers))
        inventory_cost_expr = quicksum(inventory_costs[i] * inventory_vars[f"inventory_{i+1}"] for i in range(self.n_facilities))

        objective_expr = fixed_cost_expr + transport_cost_expr + inventory_cost_expr

        # Constraints
        for j in range(self.n_customers):
            demand_constraint = quicksum(x_vars[f"x_{i+1}_{j+1}"] for i in range(self.n_facilities))
            model.addCons(demand_constraint >= customers[j]['demand'], f"demand_{j+1}")

        for i in range(self.n_facilities):
            capacity_constraint = quicksum(x_vars[f"x_{i+1}_{j+1}"] for j in range(self.n_customers))
            model.addCons(capacity_constraint <= facilities[i]['capacity'] * y_vars[f"y_{i+1}"], f"capacity_{i+1}")

            inventory_balance = quicksum(x_vars[f"x_{i+1}_{j+1}"] for j in range(self.n_customers)) - inventory_vars[f"inventory_{i+1}"]
            model.addCons(inventory_balance == 0, f"inventory_balance_{i+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_facilities': 25,
        'max_n_facilities': 500,
        'min_n_customers': 75,
        'max_n_customers': 180,
        'facility_fixed_cost_min': 2000,
        'facility_fixed_cost_max': 5000,
        'facility_capacity_min': 3000,
        'facility_capacity_max': 1500,
        'customer_demand_min': 1000,
        'customer_demand_max': 1800,
        'transport_cost_min': 5,
        'transport_cost_max': 100,
        'inventory_cost_min': 0.71,
        'inventory_cost_max': 2,
    }

    facility_planning = FacilityPlanningCostObjective(parameters, seed=seed)
    instance = facility_planning.generate_instance()
    solve_status, solve_time = facility_planning.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")