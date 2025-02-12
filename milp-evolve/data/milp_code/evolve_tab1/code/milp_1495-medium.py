import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class LogisticsDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_customers * self.n_trucks * self.demand_density)

        # compute number of customers per truck
        indices = np.random.choice(self.n_trucks, size=nnzrs)  # random truck indexes
        indices[:2 * self.n_trucks] = np.repeat(np.arange(self.n_trucks), 2)  # force at least 2 customers per truck
        _, truck_ncustomers = np.unique(indices, return_counts=True)

        # for each truck, sample random customers
        indices[:self.n_customers] = np.random.permutation(self.n_customers)  # force at least 1 truck per customer
        i = 0
        indptr = [0]
        for n in truck_ncustomers:
            if i >= self.n_customers:
                indices[i:i + n] = np.random.choice(self.n_customers, size=n, replace=False)
            elif i + n > self.n_customers:
                remaining_customers = np.setdiff1d(np.arange(self.n_customers), indices[i:self.n_customers], assume_unique=True)
                indices[self.n_customers:i + n] = np.random.choice(remaining_customers, size=i + n - self.n_customers, replace=False)
            i += n
            indptr.append(i)

        # objective coefficients (trucks' operating costs)
        c = np.random.randint(self.max_cost, size=self.n_trucks) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_customers, self.n_trucks)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        # Truck capacities and holiday bonus
        truck_capacities = np.random.randint(self.min_truck_capacity, self.max_truck_capacity + 1, size=self.n_trucks)
        holiday_bonus = np.random.randint(self.min_bonus, self.max_bonus + 1, size=self.n_trucks)
        
        res = {'c': c,
               'indptr_csr': indptr_csr,
               'indices_csr': indices_csr,
               'truck_capacities': truck_capacities,
               'holiday_bonus': holiday_bonus}

        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        truck_capacities = instance['truck_capacities']
        holiday_bonus = instance['holiday_bonus']

        model = Model("LogisticsDeliveryOptimization")

        # Create variables for truck allocation and scheduling
        var_allocation = {j: model.addVar(vtype="B", name=f"allocation_{j}", obj=c[j]) for j in range(self.n_trucks)}
        schedule_time = model.addVar(vtype="I", name="schedule_time")  # Scheduling time variable

        # Ensure each customer's demand is met
        for customer in range(self.n_customers):
            trucks = indices_csr[indptr_csr[customer]:indptr_csr[customer + 1]]
            model.addCons(quicksum(var_allocation[j] for j in trucks) >= 1, f"customer_demand_{customer}")

        # Ensure truck capacity limit and holiday bonus scheduling
        for j in range(self.n_trucks):
            model.addCons(schedule_time * var_allocation[j] <= truck_capacities[j], f"capacity_{j}")
            model.addCons(schedule_time >= holiday_bonus[j], f"holiday_bonus_{j}")
        
        # Objective: minimize total cost and holiday bonus duration
        objective_expr = quicksum(var_allocation[j] * c[j] for j in range(self.n_trucks)) + schedule_time
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 2000,
        'n_trucks': 1400,
        'demand_density': 0.1,
        'max_cost': 2000,
        'min_truck_capacity': 700,
        'max_truck_capacity': 3000,
        'min_bonus': 30,
        'max_bonus': 1200,
    }

    logistics_delivery_optimization = LogisticsDeliveryOptimization(parameters, seed=seed)
    instance = logistics_delivery_optimization.generate_instance()
    solve_status, solve_time = logistics_delivery_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")