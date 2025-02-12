import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum


class MultiItemLotSizing:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        """generate example data set"""
        setup_costs, setup_times, variable_costs, demands, holding_costs, resource_upper_bounds = {}, {}, {}, {}, {}, {}

        sumT = 0
        for t in range(1, self.num_periods+1):
            for p in range(1, self.num_products+1):
                setup_times[t, p] = 10 * random.randint(1, 5)  
                setup_costs[t, p] = 100 * random.randint(1, 10) 
                variable_costs[t, p] = 0         

                # demands
                demands[t, p] = 100 + random.randint(-25, 25) 
                if t <= 4:
                    if random.random() < 0.25:     
                        demands[t,p] = 0
                # sumT is the total capacity usage in the lot-for-lot solution
                sumT += setup_times[t,p] + demands[t,p]             
                holding_costs[t, p] = random.randint(1, 5)  

        for t in range(1, self.num_periods+1):
            resource_upper_bounds[t] = int(float(sumT) / (float(self.num_periods) * self.factor))

        res = {
            'setup_costs': setup_costs,
            'setup_times': setup_times,
            'variable_costs': variable_costs,
            'demands': demands,
            'holding_costs': holding_costs,
            'resource_upper_bounds': resource_upper_bounds
        }
        return res

    def solve(self, instance):

        setup_costs = instance['setup_costs']
        setup_times = instance['setup_times']
        variable_costs = instance['variable_costs']
        demands = instance['demands']
        holding_costs = instance['holding_costs']
        resource_upper_bounds = instance['resource_upper_bounds']

        model = Model("standard multi-item lotsizing")

        y, x, I = {}, {}, {}
        for p in range(1, self.num_products+1):
            for t in range(1, self.num_periods+1):
                y[t, p] = model.addVar(vtype="B", name="y_%s_%s" % (t,p))
                x[t, p] = model.addVar(vtype="C", name="x_%s_%s" % (t,p))
                I[t, p] = model.addVar(vtype="C", name="I_%s_%s" % (t,p))
            I[0, p] = 0

        for t in range(1, self.num_periods+1):
            # time capacity constraints
            model.addCons(quicksum(setup_times[t, p] * y[t, p] + x[t, p] 
                                   for p in range(1, self.num_products+1)) <= resource_upper_bounds[t], "TimeUB_%s" % t)

            for p in range(1, self.num_products+1):
                # flow conservation constraints
                model.addCons(I[t-1, p] + x[t,p] == I[t, p] + demands[t,p], "FlowCons_%s_%s" % (t, p))

                # capacity connection constraints
                model.addCons(x[t,p] <= (resource_upper_bounds[t] - setup_times[t,p]) * y[t, p], "ConstrUB_%s_%s" % (t, p))

                # tighten constraints
                model.addCons(x[t,p] <= demands[t,p] * y[t,p] + I[t,p], "Tighten_%s_%s" % (t, p))

        objective_expr = quicksum(setup_costs[t,p] * y[t,p] + 
                                  variable_costs[t,p] * x[t,p] + 
                                  holding_costs[t,p] * I[t,p] 
                                  for t in range(1, self.num_periods+1) for p in range(1, self.num_products+1))
        
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time



if __name__ == "__main__":
    parameters = {
        'num_periods': 30,
        'num_products': 10,
        'factor': 1.0
    }

    model = MultiItemLotSizing(parameters)
    instance = model.generate_instance()
    solve_status, solve_time = model.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
