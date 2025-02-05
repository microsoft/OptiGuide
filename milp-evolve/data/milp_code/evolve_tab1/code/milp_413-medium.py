import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HealthcareStrategicPlanning:
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
            (rand(self.m_customers, 1) - rand(1, self.m_providers))**2 +
            (rand(self.m_customers, 1) - rand(1, self.m_providers))**2
        )
        return costs

    def gamma_dist(self, size, shape, scale):
        return np.random.gamma(shape, scale, size)
    
    def generate_instance(self):
        demands = self.normal_dist(self.m_customers, self.demand_mean, self.demand_stddev).astype(int)
        capacities = self.gamma_dist(self.m_providers, self.capacity_shape, self.capacity_scale).astype(int)
        setup_costs = (
            self.normal_dist(self.m_providers, self.setup_cost_mean, self.setup_cost_stddev).astype(int) + 
            self.randint(self.m_providers, self.setup_cost_interval)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]
        home_service_budgets = self.normal_dist(self.m_providers, self.home_service_mean, self.home_service_stddev).astype(int)

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'setup_costs': setup_costs,
            'transportation_costs': transportation_costs,
            'home_service_budgets': home_service_budgets
        }
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        setup_costs = instance['setup_costs']
        transportation_costs = instance['transportation_costs']
        home_service_budgets = instance['home_service_budgets']
        
        m_customers = len(demands)
        m_providers = len(capacities)
        
        model = Model("HealthcareStrategicPlanning")
        
        # Decision variables
        healthcare_provider = {j: model.addVar(vtype="B", name=f"HealthcareProvider_{j}") for j in range(m_providers)}
        nursery = {j: model.addVar(vtype="B", name=f"Nursery_{j}") for j in range(m_providers)}
        if self.continuous_distribution:
            transport = {(i, j): model.addVar(vtype="C", name=f"Transport_{i}_{j}") for i in range(m_customers) for j in range(m_providers)}
        else:
            transport = {(i, j): model.addVar(vtype="B", name=f"Transport_{i}_{j}") for i in range(m_customers) for j in range(m_providers)}
        home_service = {j: model.addVar(vtype="C", name=f"HomeService_{j}") for j in range(m_providers)}

        # Objective: minimize the total cost
        objective_expr = quicksum(setup_costs[j] * healthcare_provider[j] for j in range(m_providers)) + \
                         quicksum(transportation_costs[i, j] * transport[i, j] for i in range(m_customers) for j in range(m_providers)) + \
                         quicksum(home_service_budgets[j] * home_service[j] for j in range(m_providers))
        
        model.setObjective(objective_expr, "minimize")
        
        # Constraints: demand must be met
        for i in range(m_customers):
            model.addCons(quicksum(transport[i, j] for j in range(m_providers)) >= 1, f"Demand_{i}")
        
        # Constraints: capacity limits
        for j in range(m_providers):
            model.addCons(quicksum(transport[i, j] * demands[i] for i in range(m_customers)) <= capacities[j] * healthcare_provider[j], f"Capacity_{j}")

        # Constraints: home service budget limits
        for j in range(m_providers):
            model.addCons(home_service[j] <= self.max_home_service_budget, f"HomeServiceBudget_{j}")
        
        # Constraints: Sales Target
        total_sales = np.sum(demands)
        model.addCons(quicksum(capacities[j] * healthcare_provider[j] for j in range(m_providers)) >= self.annual_master_plan * total_sales, "AnnualMasterPlan")
        
        # Symmetry-breaking constraints using lexicographical order
        for j in range(m_providers - 1):
            model.addCons(healthcare_provider[j] >= healthcare_provider[j + 1], f"Symmetry_{j}_{j+1}")

        # Nursery Constraints
        for j in range(m_providers):
            model.addCons(nursery[j] == 1 - healthcare_provider[j], f"Nursery_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'm_customers': 37,
        'm_providers': 250,
        'demand_mean': 225,
        'demand_stddev': 60,
        'capacity_shape': 150.0,
        'capacity_scale': 1575.0,
        'setup_cost_mean': 1200.0,
        'setup_cost_stddev': 400.0,
        'setup_cost_interval': (0, 12),
        'home_service_mean': 5000,
        'home_service_stddev': 750,
        'max_home_service_budget': 7000,
        'annual_master_plan': 0.66,
        'ratio': 13.12,
        'continuous_distribution': 0,
    }

    healthcare_planner = HealthcareStrategicPlanning(parameters, seed=seed)
    instance = healthcare_planner.generate_instance()
    solve_status, solve_time = healthcare_planner.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")