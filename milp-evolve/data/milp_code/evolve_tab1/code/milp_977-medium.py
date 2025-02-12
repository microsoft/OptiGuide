import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class AdvancedFacilityLocation:
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

        # More Complex Data Generation
        energy_consumption = np.random.uniform(0.5, 2.0, self.n_facilities).tolist()
        raw_material_availability = np.random.uniform(50, 200, self.n_neighborhoods).tolist()
        labor_cost = np.random.uniform(10, 50, self.n_facilities).tolist()
        environmental_impact = np.random.normal(20, 5, self.n_facilities).tolist()

        demand_fluctuation = np.random.normal(1, 0.2, self.n_neighborhoods).tolist()
        ordered_neighborhoods = list(np.random.permutation(self.n_neighborhoods))
        
        maintenance_schedules = np.random.choice([True, False], self.n_facilities, p=[0.1, 0.9]).tolist()
        breakdown_probabilities = np.random.uniform(0.01, 0.1, self.n_facilities).tolist()
        employee_shift_costs = np.random.uniform(20, 80, self.n_facilities).tolist()
        shift_availability = [np.random.choice([True, False], 3, p=[0.7, 0.3]).tolist() for _ in range(self.n_facilities)]
        delivery_schedule = np.random.uniform(0.7, 1.3, self.n_neighborhoods).tolist()
        
        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            fac1 = random.randint(0, self.n_facilities - 1)
            fac2 = random.randint(0, self.n_facilities - 1)
            if fac1 != fac2:
                mutual_exclusivity_pairs.append((fac1, fac2))

        seasonality = np.random.uniform(0.5, 1.5, 4).tolist()  # Quarterly variation
        
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
            "maintenance_schedules": maintenance_schedules,
            "breakdown_probabilities": breakdown_probabilities,
            "employee_shift_costs": employee_shift_costs,
            "shift_availability": shift_availability,
            "delivery_schedule": delivery_schedule,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "seasonality": seasonality,
        }

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
        maintenance_schedules = instance['maintenance_schedules']
        breakdown_probabilities = instance['breakdown_probabilities']
        employee_shift_costs = instance['employee_shift_costs']
        shift_availability = instance['shift_availability']
        delivery_schedule = instance['delivery_schedule']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        seasonality = instance['seasonality']

        model = Model("AdvancedFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        capacity_used_vars = {f: model.addVar(vtype="C", name=f"CapacityUsed_{f}", lb=0) for f in range(n_facilities)}

        seasonal_vars = {(f, q): model.addVar(vtype="C", name=f"SeasonalImpact_{f}_Q{q}", lb=0) for f in range(n_facilities) for q in range(4)}
        
        ### New variables and constraints code section
        maintenance_penalty_vars = {f: model.addVar(vtype="C", name=f"MaintenancePenalty_{f}", lb=0) for f in range(n_facilities)}

        # Unified Objective Function
        model.setObjective(
            quicksum(financial_rewards[n] * allocation_vars[f, n] * demand_fluctuation[n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(capacity_used_vars[f] * energy_consumption[f] for f in range(n_facilities)) -
            quicksum(labor_cost[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(environmental_impact[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(maintenance_penalty_vars[f] * breakdown_probabilities[f] for f in range(n_facilities)) -
            quicksum(seasonal_vars[f, q] * seasonality[q] for f in range(n_facilities) for q in range(4)),
            "maximize"
        )

        # Constraints
        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")
        
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Service_{n}")
        
        for f in range(n_facilities):
            model.addCons(capacity_used_vars[f] == quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)), f"CapacityUsage_{f}")
        
        for f in range(n_facilities):
            model.addCons(capacity_used_vars[f] <= capacities[f], f"CapacityLimit_{f}")

        for f in range(n_facilities):
            if maintenance_schedules[f]:
                model.addCons(maintenance_penalty_vars[f] >= 1, f"MaintenancePenalty_{f}")

        for n in range(n_neighborhoods):
            model.addCons(raw_material_availability[n] >= quicksum(allocation_vars[f, n] * delivery_schedule[n] for f in range(n_facilities)), f"RawMaterial_{n}")

        for f in range(n_facilities):
            model.addCons(quicksum(seasonal_vars[f, q] * seasonality[q] for q in range(4)) <= quicksum(allocation_vars[f, n] * demand_fluctuation[n] for n in range(n_neighborhoods)), f"SeasonalImpact_{f}")

        for i, (fac1, fac2) in enumerate(mutual_exclusivity_pairs):
            model.addCons(facility_vars[fac1] + facility_vars[fac2] <= 1, f"MutualExclusivity_{fac1}_{fac2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 27,
        'n_neighborhoods': 168,
        'min_fixed_cost': 562,
        'max_fixed_cost': 2108,
        'min_transport_cost': 607,
        'max_transport_cost': 2952,
        'min_capacity': 519,
        'max_capacity': 2024,
        'n_exclusive_pairs': 22,
    }

    location_optimizer = AdvancedFacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")