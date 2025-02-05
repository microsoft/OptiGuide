import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class CapacitatedFacilityLocationWithAuction:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2 +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_facilities, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_facilities, self.fixed_cost_cste_interval)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        # Generate transportation contracts based on combinatorial auction approach
        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_facilities)
        add_item_prob = self.add_item_prob
        value_deviation = self.value_deviation

        # item compatibilities
        compats = np.triu(np.random.rand(self.n_facilities, self.n_facilities), k=1)
        compats = compats + compats.transpose()
        compats = compats / compats.sum(1)

        transportation_contracts = []
        while len(transportation_contracts) < self.n_contracts:
            private_interests = np.random.rand(self.n_facilities)
            private_values = values + self.max_value * value_deviation * (2 * private_interests - 1)
            prob = private_interests / private_interests.sum()
            item = np.random.choice(self.n_facilities, p=prob)
            bundle_mask = np.full(self.n_facilities, 0)
            bundle_mask[item] = 1

            while np.random.rand() < add_item_prob:
                if bundle_mask.sum() == self.n_facilities:
                    break
                prob = (1 - bundle_mask) * private_interests * compats[bundle_mask, :].mean(axis=0)
                prob = prob / prob.sum()
                item = np.random.choice(self.n_facilities, p=prob)
                bundle_mask[item] = 1

            bundle = np.nonzero(bundle_mask)[0]
            price = private_values[bundle].sum() + np.power(len(bundle), 1 + self.additivity)
            transportation_contracts.append((list(bundle), price))

        res['transportation_contracts'] = transportation_contracts

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("CapacitatedFacilityLocationWithAuction")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        if self.continuous_assignment:
            serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        else:
            serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}

        # New decision variable for transportation contracts
        transportation_contracts = instance['transportation_contracts']
        contract_vars = {c: model.addVar(vtype="B", name=f"Contract_{c}") for c in range(len(transportation_contracts))}

        # Objective: minimize the total cost
        objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        objective_expr += quicksum(price * contract_vars[c] for c, (bundle, price) in enumerate(transportation_contracts))
        
        # Constraints: demand must be met
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}")
        
        # Constraints: capacity limits
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")
        
        # Constraints: tightening constraints
        total_demand = np.sum(demands)
        model.addCons(quicksum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_demand, "TotalDemand")
        
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(serve[i, j] <= open_facilities[j], f"Tightening_{i}_{j}")

        # Additional constraints for transportation contracts
        for c, (bundle, price) in enumerate(transportation_contracts):
            model.addCons(quicksum(open_facilities[j] for j in bundle) >= len(bundle) * contract_vars[c], f"Contract_{c}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 200,
        'n_facilities': 100,
        'demand_interval': (50, 360),
        'capacity_interval': (20, 322),
        'fixed_cost_scale_interval': (700, 777),
        'fixed_cost_cste_interval': (0, 45),
        'ratio': 45.0,
        'continuous_assignment': 9,
        'n_contracts': 3000,
        'min_value': 1,
        'max_value': 1000,
        'value_deviation': 0.59,
        'additivity': 0.24,
        'add_item_prob': 0.66,
    }

    facility_location = CapacitatedFacilityLocationWithAuction(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")