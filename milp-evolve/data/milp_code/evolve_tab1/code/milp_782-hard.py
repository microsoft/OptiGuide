import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_neighborhoods > 0
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_neighborhoods))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)

        financial_rewards = np.random.uniform(10, 100, self.n_neighborhoods)
        temp_control_costs = np.random.uniform(5.0, 15.0, (self.n_facilities, self.n_neighborhoods))
        max_temp = 8  # Maximum allowable temperature in each vehicle

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "financial_rewards": financial_rewards,
            "temp_control_costs": temp_control_costs,
            "max_temp": max_temp
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        financial_rewards = instance['financial_rewards']
        temp_control_costs = instance['temp_control_costs']
        max_temp = instance['max_temp']
        
        model = Model("ComplexFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        healthcare_provider_vars = {h: model.addVar(vtype="B", name=f"HealthcareProvider_{h}") for h in range(self.n_healthcare_providers)}
        temperature_control_vars = {(f, n): model.addVar(vtype="B", name=f"TempControl_{f}_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        
        # Objective: maximize financial rewards from treated neighborhoods minus costs (fixed, transport, temp control)
        model.setObjective(
            quicksum(financial_rewards[n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(temp_control_costs[f][n] * temperature_control_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)),
            "maximize"
        )

        # Constraints: Each neighborhood is served by exactly one facility
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")
        
        # Constraints: Only open facilities can serve neighborhoods
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Service_{n}")
        
        # Constraints: Facilities cannot exceed their capacity
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_Capacity")
        
        # New Constraints: Each healthcare provider can work at exactly one facility
        for h in range(self.n_healthcare_providers):
            model.addCons(quicksum(healthcare_provider_vars[h] * facility_vars[f] for f in range(n_facilities)) <= 1, f"Healthcare_{h}_Assignment")
        
        # New Constraints: Temperature control must be within maximum limits
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(temperature_control_vars[f, n] * max_temp >= self.min_temp, f"TemperatureControl_{f}_{n}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 25,
        'n_neighborhoods': 200,
        'min_fixed_cost': 1500,
        'max_fixed_cost': 1500,
        'min_transport_cost': 450,
        'max_transport_cost': 1500,
        'min_capacity': 49,
        'max_capacity': 1400,
        'n_healthcare_providers': 22,
        'min_temp': 2,
    }

    location_optimizer = FacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")