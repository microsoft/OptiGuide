import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SimplifiedFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_neighborhoods >= self.n_facilities
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        
        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_neighborhoods))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)
        financial_rewards = np.random.uniform(10, 100, self.n_neighborhoods)
        demand_fluctuation = np.random.normal(1, 0.2, self.n_neighborhoods).tolist()
        ordered_neighborhoods = list(np.random.permutation(self.n_neighborhoods))
        
        priority_rewards = np.random.uniform(50, 200, self.n_neighborhoods)
        priority_levels = np.random.choice([0, 1], size=self.n_neighborhoods, p=[0.7, 0.3])  # 30% neighborhoods are high priority
        
        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "financial_rewards": financial_rewards,
            "demand_fluctuation": demand_fluctuation,
            "ordered_neighborhoods": ordered_neighborhoods,
            "priority_rewards": priority_rewards,
            "priority_levels": priority_levels,
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        financial_rewards = instance['financial_rewards']
        demand_fluctuation = instance['demand_fluctuation']
        ordered_neighborhoods = instance['ordered_neighborhoods']
        priority_rewards = instance['priority_rewards']
        priority_levels = instance['priority_levels']

        model = Model("SimplifiedFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}

        priority_vars = {n: model.addVar(vtype="I", lb=0, ub=1, name=f"Priority_{n}") for n in range(n_neighborhoods)}
        service_vars = {(f, n): model.addVar(vtype="C", name=f"Service_{f}_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        
        model.setObjective(
            quicksum(financial_rewards[n] * allocation_vars[f, n] * demand_fluctuation[n] for f in range(n_facilities) for n in range(n_neighborhoods)) +
            quicksum(priority_rewards[n] * priority_vars[n] for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * service_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)),
            "maximize"
        )

        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")
        
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Service_{n}")
        
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_Capacity")

        for n in range(n_neighborhoods):
            for f in range(n_facilities):
                model.addCons(allocation_vars[f, n] * demand_fluctuation[n] <= capacities[f], f"DemandCapacity_{f}_{n}")

        for i in range(n_neighborhoods - 1):
            n1 = ordered_neighborhoods[i]
            n2 = ordered_neighborhoods[i + 1]
            for f in range(n_facilities):
                model.addCons(allocation_vars[f, n1] + allocation_vars[f, n2] <= 1, f"SOS_Constraint_Facility_{f}_Neighborhoods_{n1}_{n2}")

        # Big M constraints for priority levels and service levels
        M = max(self.max_capacity, sum(financial_rewards))  # Big M value

        for n in range(n_neighborhoods):
            model.addCons(priority_vars[n] >= priority_levels[n], f"PriorityLevel_{n}")
            model.addCons(priority_vars[n] <= quicksum(allocation_vars[f, n] for f in range(n_facilities)), f"PriorityService_{n}")

            for f in range(n_facilities):
                model.addCons(service_vars[f, n] >= allocation_vars[f, n], f"ServiceMin_{f}_{n}")
                model.addCons(service_vars[f, n] <= M * allocation_vars[f, n], f"ServiceMax_{f}_{n}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 39,
        'n_neighborhoods': 252,
        'min_fixed_cost': 187,
        'max_fixed_cost': 1185,
        'min_transport_cost': 121,
        'max_transport_cost': 686,
        'min_capacity': 2380,
        'max_capacity': 2530,
    }

    parameters.update({
        'min_service_level': 10,
        'max_service_level': 1000,
        'big_M': 2000,
    })

    location_optimizer = SimplifiedFacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")