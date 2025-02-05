import random
import time
import numpy as np
from itertools import product
from pyscipopt import Model, quicksum

############# Helper function #############
# Helper class definitions can be added here if needed
# In this case, we do not have a separate helper class.

class LogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        # Generate fixed opening costs of facilities
        FacilityCost = np.random.randint(1000, 5000, self.n_facilities)
        
        # Generate variable operating costs of facilities
        FacilityOperatingCost = np.random.randint(10, 50, self.n_facilities)
        
        # Generate transportation costs between facilities and customers
        TransportCost = np.random.randint(1, 10, (self.n_facilities, self.n_customers))
        
        # Generate facility capacities
        FacilityCapacity = np.random.randint(50, 100, self.n_facilities)
        
        # Generate customer demands
        CustomerDemand = np.random.randint(1, 30, self.n_customers)
        
        res = {
            'FacilityCost': FacilityCost,
            'FacilityOperatingCost': FacilityOperatingCost,
            'TransportCost': TransportCost,
            'FacilityCapacity': FacilityCapacity,
            'CustomerDemand': CustomerDemand
        }
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        FacilityCost = instance['FacilityCost']
        FacilityOperatingCost = instance['FacilityOperatingCost']
        TransportCost = instance['TransportCost']
        FacilityCapacity = instance['FacilityCapacity']
        CustomerDemand = instance['CustomerDemand']

        model = Model("LogisticsOptimization")
        x = {}
        y = {}
        z = {}

        # Create binary variables for facility opening decision
        for f in range(self.n_facilities):
            x[f] = model.addVar(vtype="B", name=f"x_{f}")

        # Create continuous variables for transportation quantities
        for f, c in product(range(self.n_facilities), range(self.n_customers)):
            y[f, c] = model.addVar(vtype="C", name=f"y_{f}_{c}")

        # Create binary variables for customer assignments
        for f, c in product(range(self.n_facilities), range(self.n_customers)):
            z[f, c] = model.addVar(vtype="B", name=f"z_{f}_{c}")

        # Add capacity constraint for facilities
        for f in range(self.n_facilities):
            model.addCons(
                quicksum(y[f, c] for c in range(self.n_customers)) <= FacilityCapacity[f] * x[f],
                name=f"FacilityCapacity_{f}"
            )

        # Add demand satisfaction constraint for customers
        for c in range(self.n_customers):
            model.addCons(
                quicksum(y[f, c] for f in range(self.n_facilities)) == CustomerDemand[c],
                name=f"CustomerDemand_{c}"
            )

        # Add constraint linking binary assignment and transportation variables
        for f, c in product(range(self.n_facilities), range(self.n_customers)):
            model.addCons(
                y[f, c] <= CustomerDemand[c] * z[f, c],
                name=f"Link_{f}_{c}"
            )
            model.addCons(
                z[f, c] <= x[f],
                name=f"Assignment_{f}_{c}"
            )

        # Define the objective to minimize total cost
        model.setObjective(
            quicksum(FacilityCost[f] * x[f] + FacilityOperatingCost[f] * quicksum(y[f, c] for c in range(self.n_customers)) + TransportCost[f, c] * y[f, c] for f, c in product(range(self.n_facilities), range(self.n_customers))),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 100,
        'n_customers': 400,
    }

    logistics_optimization = LogisticsOptimization(parameters, seed=seed)
    instance = logistics_optimization.generate_instance()
    solve_status, solve_time = logistics_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")