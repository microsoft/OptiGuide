import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ProductionScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        jobs = np.random.randint(self.min_job_demand, self.max_job_demand, self.number_of_jobs)
        job_deadlines = np.random.randint(self.min_job_deadline, self.max_job_deadline, self.number_of_jobs)
        job_costs = np.random.randint(self.min_job_cost, self.max_job_cost, self.number_of_jobs)
        
        factory_capacities = np.random.randint(self.min_factory_capacity, self.max_factory_capacity, self.number_of_factories)
        factory_resource_limits = np.random.randint(self.min_resource_limit, self.max_resource_limit, (self.number_of_factories, self.number_of_resources))
        operation_costs = np.random.randint(self.min_operation_cost, self.max_operation_cost, self.number_of_factories)
        
        resource_requirements = np.random.randint(self.min_resource_requirement, self.max_resource_requirement, (self.number_of_jobs, self.number_of_resources))
        res = {
            'jobs': jobs, 
            'job_deadlines': job_deadlines,
            'job_costs': job_costs,
            'factory_capacities': factory_capacities,
            'factory_resource_limits': factory_resource_limits,
            'operation_costs': operation_costs,
            'resource_requirements': resource_requirements
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        jobs = instance['jobs']
        job_deadlines = instance['job_deadlines']
        job_costs = instance['job_costs']
        factory_capacities = instance['factory_capacities']
        factory_resource_limits = instance['factory_resource_limits']
        operation_costs = instance['operation_costs']
        resource_requirements = instance['resource_requirements']
        
        num_jobs = len(jobs)
        num_factories = len(factory_capacities)
        num_resources = self.number_of_resources
        num_periods = self.number_of_periods
        
        model = Model("ProductionScheduling")
        var_names = {}
        
        # Decision variables: x[f][j][t] = 1 if job j is assigned to factory f at time t
        for f in range(num_factories):
            for j in range(num_jobs):
                for t in range(num_periods):
                    var_names[(f, j, t)] = model.addVar(vtype="B", name=f"x_{f}_{j}_{t}")
                    
        # Decision variables: y[f][t] = 1 if factory f operates at time t
        factory_operate = {}
        for f in range(num_factories):
            for t in range(num_periods):
                factory_operate[(f, t)] = model.addVar(vtype="B", name=f"y_{f}_{t}")
        
        # Objective: Minimize total cost
        objective_expr = quicksum(job_costs[j] * var_names[(f, j, t)]
                                  for f in range(num_factories)
                                  for j in range(num_jobs)
                                  for t in range(num_periods))
        # Adding operation cost to objective
        cost_expr = quicksum(operation_costs[f] * factory_operate[(f, t)]
                             for f in range(num_factories)
                             for t in range(num_periods))
        
        model.setObjective(objective_expr + cost_expr, "minimize")
        
        # Constraints: Each job must be completed by its deadline
        for j in range(num_jobs):
            model.addCons(
                quicksum(var_names[(f, j, t)] 
                         for f in range(num_factories) 
                         for t in range(job_deadlines[j])) >= 1,
                f"CompletingJob_{j}"
            )
        
        # Constraints: Total jobs in each factory must not exceed its capacity at any time
        for f in range(num_factories):
            for t in range(num_periods):
                model.addCons(
                    quicksum(var_names[(f, j, t)] * jobs[j] for j in range(num_jobs)) <= factory_capacities[f] * factory_operate[(f, t)],
                    f"FactoryCapacity_{f}_{t}"
                )
        
        # Constraints: Total resource requirements must be within limits
        for f in range(num_factories):
            for t in range(num_periods):
                for r in range(num_resources):
                    model.addCons(
                        quicksum(var_names[(f, j, t)] * resource_requirements[j][r] for j in range(num_jobs)) <= factory_resource_limits[f][r],
                        f"ResourceLimit_{f}_{t}_{r}"
                    )
            
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_jobs': 100,
        'number_of_factories': 5,
        'number_of_resources': 3,
        'number_of_periods': 10,
        'min_job_demand': 5,
        'max_job_demand': 20,
        'min_job_deadline': 3,
        'max_job_deadline': 10,
        'min_job_cost': 50,
        'max_job_cost': 200,
        'min_factory_capacity': 50,
        'max_factory_capacity': 100,
        'min_resource_limit': 100,
        'max_resource_limit': 300,
        'min_operation_cost': 100,
        'max_operation_cost': 500,
        'min_resource_requirement': 1,
        'max_resource_requirement': 10,
    }

    scheduling = ProductionScheduling(parameters, seed=seed)
    instance = scheduling.generate_instance()
    solve_status, solve_time = scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")