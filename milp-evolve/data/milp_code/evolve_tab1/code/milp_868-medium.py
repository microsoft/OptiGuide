import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ComplexFacilityLocation:
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

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_neighborhoods))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)

        financial_rewards = np.random.uniform(10, 100, self.n_neighborhoods)

        # Additional data generation
        energy_consumption = np.random.uniform(0.5, 2.0, self.n_facilities).tolist()
        raw_material_availability = np.random.uniform(50, 200, self.n_neighborhoods).tolist()
        labor_cost = np.random.uniform(10, 50, self.n_facilities).tolist()
        environmental_impact = np.random.normal(20, 5, self.n_facilities).tolist()

        # New data generation: Fluctuating market demands
        demand_fluctuation = np.random.normal(1, 0.2, self.n_neighborhoods).tolist()

        # Special Ordered Sets (SOS) data
        ordered_neighborhoods = list(np.random.permutation(self.n_neighborhoods))

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "financial_rewards": financial_rewards,
            "energy_consumption": energy_consumption,
            "raw_material_availability": raw_material_availability,
            "labor_cost": labor_cost,
            "environmental_impact": environmental_impact,
            "demand_fluctuation": demand_fluctuation,
            "ordered_neighborhoods": ordered_neighborhoods,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        financial_rewards = instance['financial_rewards']
        energy_consumption = instance['energy_consumption']
        raw_material_availability = instance['raw_material_availability']
        labor_cost = instance['labor_cost']
        environmental_impact = instance['environmental_impact']
        demand_fluctuation = instance['demand_fluctuation']
        ordered_neighborhoods = instance['ordered_neighborhoods']

        model = Model("ComplexFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}

        # New variables
        energy_vars = {f: model.addVar(vtype="C", name=f"Energy_{f}", lb=0) for f in range(n_facilities)}
        raw_material_vars = {n: model.addVar(vtype="C", name=f"RawMaterial_{n}", lb=0) for n in range(n_neighborhoods)}
        labor_cost_vars = {f: model.addVar(vtype="C", name=f"LaborCost_{f}", lb=0) for f in range(n_facilities)}
        environmental_impact_vars = {f: model.addVar(vtype="C", name=f"EnvironmentalImpact_{f}", lb=0) for f in range(n_facilities)}
        pricing_vars = {n: model.addVar(vtype="C", name=f"Pricing_{n}", lb=0) for n in range(n_neighborhoods)}

        # Objective: maximize financial rewards from treated neighborhoods minus costs (fixed and transport) and additional costs
        model.setObjective(
            quicksum(financial_rewards[n] * allocation_vars[f, n] * demand_fluctuation[n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(energy_vars[f] * energy_consumption[f] for f in range(n_facilities)) -
            quicksum(labor_cost_vars[f] * labor_cost[f] for f in range(n_facilities)) -
            quicksum(environmental_impact_vars[f] * environmental_impact[f] for f in range(n_facilities)),
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

        # Additional constraints
        for f in range(n_facilities):
            model.addCons(energy_vars[f] == quicksum(allocation_vars[f, n] * energy_consumption[f] for n in range(n_neighborhoods)), f"EnergyConsumption_{f}")

        for n in range(n_neighborhoods):
            model.addCons(raw_material_vars[n] <= raw_material_availability[n], f"RawMaterial_{n}")

        for f in range(n_facilities):
            model.addCons(labor_cost_vars[f] <= labor_cost[f], f"LaborCost_{f}")

        for f in range(n_facilities):
            model.addCons(environmental_impact_vars[f] <= environmental_impact[f], f"EnvironmentalImpact_{f}")

        # New constraints: Pricing based on demand fluctuation and financial rewards
        for n in range(n_neighborhoods):
            model.addCons(pricing_vars[n] == financial_rewards[n] * demand_fluctuation[n], f"Pricing_{n}")

        # Inter-dependency constraints (coupling facilities and neighborhoods)
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] * demand_fluctuation[n] <= capacities[f], f"DemandCapacity_{f}_{n}")

        # Special Ordered Sets (SOS) Constraints
        for i in range(n_neighborhoods - 1):
            n1 = ordered_neighborhoods[i]
            n2 = ordered_neighborhoods[i + 1]
            for f in range(n_facilities):
                model.addCons(allocation_vars[f, n1] + allocation_vars[f, n2] <= 1, f"SOS_Constraint_Facility_{f}_Neighborhoods_{n1}_{n2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 36,
        'n_neighborhoods': 225,
        'min_fixed_cost': 750,
        'max_fixed_cost': 2811,
        'min_transport_cost': 405,
        'max_transport_cost': 2625,
        'min_capacity': 924,
        'max_capacity': 1800,
        'energy_min': 0.45,
        'energy_max': 18.0,
        'raw_material_min': 375,
        'raw_material_max': 1400,
        'labor_cost_min': 350,
        'labor_cost_max': 36,
        'environmental_mean': 500,
        'environmental_std': 2025,
    }
    ### new parameter code starts here
    parameters.update({
        'demand_avg': 1,
        'demand_std': 0.2,
    })
    ### new parameter code ends here

    location_optimizer = ComplexFacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")