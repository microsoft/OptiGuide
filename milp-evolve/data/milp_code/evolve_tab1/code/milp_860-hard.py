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
        assert self.n_facilities > 0 and self.n_neighborhoods >= self.n_facilities
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        assert self.min_environmental_impact >= 0 and self.max_environmental_impact >= self.min_environmental_impact

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.exponential(self.mean_transport_cost, (self.n_facilities, self.n_neighborhoods)).astype(int)
        transport_costs = np.clip(transport_costs, self.min_transport_cost, self.max_transport_cost)
        capacities = np.random.poisson(lam=(self.min_capacity + self.max_capacity) / 2, size=self.n_facilities)
        environmental_impacts = np.random.uniform(self.min_environmental_impact, self.max_environmental_impact, self.n_facilities)
        social_benefits = np.random.uniform(10, 100, self.n_neighborhoods)
        
        facility_types = np.random.choice(['A', 'B'], size=self.n_facilities, p=[0.6, 0.4])
        operating_costs = {'A': 100, 'B': 200}
        min_service_levels = np.random.randint(self.min_service_level, self.max_service_level + 1, self.n_facilities)

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "environmental_impacts": environmental_impacts,
            "social_benefits": social_benefits,
            "facility_types": facility_types,
            "operating_costs": operating_costs,
            "min_service_levels": min_service_levels,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        environmental_impacts = instance['environmental_impacts']
        social_benefits = instance['social_benefits']
        facility_types = instance['facility_types']
        operating_costs = instance['operating_costs']
        min_service_levels = instance['min_service_levels']
        
        model = Model("AdvancedFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        
        # Objective: maximize social benefits from treated neighborhoods minus costs (fixed, transport, and operating) minus environmental impact
        model.setObjective(
            quicksum(social_benefits[n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(operating_costs[facility_types[f]] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(environmental_impacts[f] * facility_vars[f] for f in range(n_facilities)),
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
        
        # Constraints: Facilities must meet their minimum service level
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] * social_benefits[n] for n in range(n_neighborhoods)) >= min_service_levels[f] * facility_vars[f], f"Facility_{f}_MinServiceLevel")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 9,
        'n_neighborhoods': 1500,
        'min_fixed_cost': 1125,
        'max_fixed_cost': 1405,
        'mean_transport_cost': 2100,
        'stddev_transport_cost': 1125,
        'min_transport_cost': 2024,
        'max_transport_cost': 2250,
        'min_capacity': 440,
        'max_capacity': 600,
        'min_service_level': 350,
        'max_service_level': 3000,
        'min_environmental_impact': 75,
        'max_environmental_impact': 2700,
    }
    
    location_optimizer = FacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")