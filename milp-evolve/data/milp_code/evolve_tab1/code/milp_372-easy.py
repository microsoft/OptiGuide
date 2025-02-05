import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class RetailChainLocationPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def normal_dist(self, size, mean, stddev):
        return np.random.normal(mean, stddev, size)

    def unit_transportation_costs(self):
        scaling = 15.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.m_customers, 1) - rand(1, self.m_facilities))**2 +
            (rand(self.m_customers, 1) - rand(1, self.m_facilities))**2
        )
        return costs

    def gamma_dist(self, size, shape, scale):
        return np.random.gamma(shape, scale, size)
    
    def generate_instance(self):
        demands = self.normal_dist(self.m_customers, self.demand_mean, self.demand_stddev).astype(int)
        capacities = self.gamma_dist(self.m_facilities, self.capacity_shape, self.capacity_scale).astype(int)
        fixed_costs = (
            self.normal_dist(self.m_facilities, self.fixed_cost_mean, self.fixed_cost_stddev).astype(int) + 
            self.randint(self.m_facilities, self.fixed_cost_cste_interval)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]
        advertising_budgets = self.normal_dist(self.m_facilities, self.advertising_mean, self.advertising_stddev).astype(int)

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'advertising_budgets': advertising_budgets
        }
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        advertising_budgets = instance['advertising_budgets']
        
        m_customers = len(demands)
        m_facilities = len(capacities)
        
        model = Model("RetailChainLocationPlanning")
        
        # Decision variables
        new_store_location = {j: model.addVar(vtype="B", name=f"NewStore_{j}") for j in range(m_facilities)}
        facility_closed = {j: model.addVar(vtype="B", name=f"Closed_{j}") for j in range(m_facilities)}
        if self.continuous_distribution:
            transport = {(i, j): model.addVar(vtype="C", name=f"Transport_{i}_{j}") for i in range(m_customers) for j in range(m_facilities)}
        else:
            transport = {(i, j): model.addVar(vtype="B", name=f"Transport_{i}_{j}") for i in range(m_customers) for j in range(m_facilities)}
        ad_budget = {j: model.addVar(vtype="C", name=f"AdBudget_{j}") for j in range(m_facilities)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * new_store_location[j] for j in range(m_facilities)) + \
                         quicksum(transportation_costs[i, j] * transport[i, j] for i in range(m_customers) for j in range(m_facilities)) + \
                         quicksum(advertising_budgets[j] * ad_budget[j] for j in range(m_facilities))
        
        model.setObjective(objective_expr, "minimize")
        
        # Constraints: demand must be met
        for i in range(m_customers):
            model.addCons(quicksum(transport[i, j] for j in range(m_facilities)) >= 1, f"Demand_{i}")
        
        # Constraints: capacity limits
        for j in range(m_facilities):
            model.addCons(quicksum(transport[i, j] * demands[i] for i in range(m_customers)) <= capacities[j] * new_store_location[j], f"Capacity_{j}")

        # Constraints: advertising budget limits
        for j in range(m_facilities):
            model.addCons(ad_budget[j] <= self.max_ad_budget, f"AdBudget_{j}")
        
        # Constraints: Sales Target
        total_sales = np.sum(demands)
        model.addCons(quicksum(capacities[j] * new_store_location[j] for j in range(m_facilities)) >= self.annual_sales_target * total_sales, "AnnualSalesTarget")
        
        # Symmetry-breaking constraints
        for j in range(m_facilities):
            for k in range(j+1, m_facilities):
                if fixed_costs[j] == fixed_costs[k]:
                    model.addCons(new_store_location[j] >= new_store_location[k], f"Symmetry_{j}_{k}")

        # Facility Closed Constraints
        for j in range(m_facilities):
            model.addCons(facility_closed[j] == 1 - new_store_location[j], f"FacilityClosed_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'm_customers': 75,
        'm_facilities': 250,
        'demand_mean': 300,
        'demand_stddev': 20,
        'capacity_shape': 10.0,
        'capacity_scale': 175.0,
        'fixed_cost_mean': 1200.0,
        'fixed_cost_stddev': 40.0,
        'fixed_cost_cste_interval': (0, 25),
        'advertising_mean': 5000,
        'advertising_stddev': 1000,
        'max_ad_budget': 7000,
        'annual_sales_target': 0.9,
        'ratio': 17.5,
        'continuous_distribution': 9,
    }

    retail_chain = RetailChainLocationPlanning(parameters, seed=seed)
    instance = retail_chain.generate_instance()
    solve_status, solve_time = retail_chain.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")