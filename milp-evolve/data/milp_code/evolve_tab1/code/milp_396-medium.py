import random
import time
import numpy as np
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

        # New instance data for piecewise linear segments
        res.update({
            'advertising_cost_segments': [self.advertising_cost_segments1, self.advertising_cost_segments2, self.advertising_cost_segments3],
            'transportation_cost_segments': [self.transportation_cost_segments1, self.transportation_cost_segments2, self.transportation_cost_segments3],
            'transportation_cost_breakpoints': self.transportation_cost_breakpoints,
            'advertising_cost_breakpoints': self.advertising_cost_breakpoints
        })

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        advertising_budgets = instance['advertising_budgets']
        advertising_cost_segments = instance['advertising_cost_segments']
        transportation_cost_segments = instance['transportation_cost_segments']
        transportation_cost_breakpoints = instance['transportation_cost_breakpoints']
        advertising_cost_breakpoints = instance['advertising_cost_breakpoints']
        
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

        # Piecewise variables for advertising costs
        ad_cost_segment_var = {(s, j): model.addVar(vtype="C", name=f"AdCostSegment_{s}_{j}") for s in range(3) for j in range(m_facilities)}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * new_store_location[j] for j in range(m_facilities)) + \
                         quicksum(transportation_costs[i, j] * transport[i, j] for i in range(m_customers) for j in range(m_facilities)) + \
                         quicksum(ad_cost_segment_var[s, j] for s in range(3) for j in range(m_facilities))
        
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

        # Piecewise linear constraints for advertising costs
        for j in range(m_facilities):
            for s in range(2):
                model.addCons(ad_cost_segment_var[s, j] <= advertising_cost_segments[s])
            model.addCons(quicksum(ad_cost_segment_var[s, j] for s in range(3)) == ad_budget[j])

        # Piecewise linear constraints for transportation costs
        piecewise_cost_vars = {(s, i, j): model.addVar(vtype="C", name=f"PieceCost_{s}_{i}_{j}") for s in range(3) for i in range(m_customers) for j in range(m_facilities)}
        for i in range(m_customers):
            for j in range(m_facilities):
                for s in range(2):
                    model.addCons(piecewise_cost_vars[s, i, j] <= transportation_cost_segments[s])
                model.addCons(quicksum(piecewise_cost_vars[s, i, j] for s in range(3)) == transport[i, j])

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'm_customers': 37,
        'm_facilities': 500,
        'demand_mean': 3000,
        'demand_stddev': 40,
        'capacity_shape': 70.0,
        'capacity_scale': 1225.0,
        'fixed_cost_mean': 900.0,
        'fixed_cost_stddev': 360.0,
        'fixed_cost_cste_interval': (0, 12),
        'advertising_mean': 5000,
        'advertising_stddev': 500,
        'max_ad_budget': 7000,
        'annual_sales_target': 0.17,
        'ratio': 8.75,
        'continuous_distribution': 27,
        'advertising_cost_segments1': 4000,
        'advertising_cost_segments2': 6000,
        'advertising_cost_segments3': 10000,
        'transportation_cost_segments1': 750,
        'transportation_cost_segments2': 1000,
        'transportation_cost_segments3': 25,
        'transportation_cost_breakpoints': (1000, 2000, 3000),
        'advertising_cost_breakpoints': (750, 3000, 4500),
    }

    retail_chain = RetailChainLocationPlanning(parameters, seed=seed)
    instance = retail_chain.generate_instance()
    solve_status, solve_time = retail_chain.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")