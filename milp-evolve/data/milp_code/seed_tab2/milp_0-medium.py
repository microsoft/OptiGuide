import math
import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class KMedianProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def distance(x1, y1, x2, y2):
        """return distance of two points"""
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def generate_instance(self):
        """generate example data set"""
        xpos = [random.random() for _ in range(self.num_customers + self.num_facilities)]
        ypos = [random.random() for _ in range(self.num_customers + self.num_facilities)]

        cost = {}
        for i in range(self.num_customers):
            for j in range(self.num_facilities):
                cost[i, j] = self.distance(xpos[i], ypos[i], xpos[j], ypos[j])

        res = {'cost': cost}
        return res

    def solve(self, instance):
        cost = instance['cost']
        
        model = Model("k-median")
        x, y = {}, {}

        for j in range(self.num_facilities):
            y[j] = model.addVar(vtype="B", name="y_%s" % j)
            for i in range(self.num_customers):
                x[i, j] = model.addVar(vtype="B", name="x_%s_%s" % (i, j))

        for i in range(self.num_customers):
            model.addCons(quicksum(x[i, j] for j in range(self.num_facilities)) == 1, "Assign_%s" % i)
            for j in range(self.num_facilities):
                model.addCons(x[i, j] <= y[j], "Strong_%s_%s" % (i, j))
        model.addCons(quicksum(y[j] for j in range(self.num_facilities)) == self.k, "Facilities")

        objective_expr = quicksum(cost[i, j] * x[i, j] for i in range(self.num_customers) for j in range(self.num_facilities))
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_customers': 300,
        'num_facilities': 300,
        'k': 20
    }

    k_median_problem = KMedianProblem(parameters, seed=seed)
    instance = k_median_problem.generate_instance()
    solve_status, solve_time = k_median_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")