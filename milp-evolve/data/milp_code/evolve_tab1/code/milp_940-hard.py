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
    
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_neighborhoods >= self.n_facilities
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        
        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_neighborhoods))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)

        financial_rewards = np.random.uniform(10, 100, self.n_neighborhoods)

        energy_consumption = np.random.uniform(0.5, 2.0, self.n_facilities).tolist()
        raw_material_availability = np.random.uniform(50, 200, self.n_neighborhoods).tolist()
        labor_cost = np.random.uniform(10, 50, self.n_facilities).tolist()
        environmental_impact = np.random.normal(20, 5, self.n_facilities).tolist()

        demand_fluctuation = np.random.normal(1, 0.2, self.n_neighborhoods).tolist()
        ordered_neighborhoods = list(np.random.permutation(self.n_neighborhoods))
        
        # New data generation: Machine maintenance schedules and breakdown probabilities
        machine_maintenance_schedules = np.random.choice([True, False], self.n_facilities, p=[0.1, 0.9]).tolist()
        machine_breakdown_probabilities = np.random.uniform(0.01, 0.1, self.n_facilities).tolist()
        
        # New data generation: Employee shift costs and availability
        employee_shift_costs = np.random.uniform(20, 80, self.n_facilities).tolist()
        employee_shift_availability = [np.random.choice([True, False], 3, p=[0.7, 0.3]).tolist() for _ in range(self.n_facilities)]
        
        # New data generation: Raw material delivery schedules
        raw_material_delivery_schedule = np.random.uniform(0.7, 1.3, self.n_neighborhoods).tolist()
        
        # Facility relations: Mutual exclusivity pairs
        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            fac1 = random.randint(0, self.n_facilities - 1)
            fac2 = random.randint(0, self.n_facilities - 1)
            if fac1 != fac2:
                mutual_exclusivity_pairs.append((fac1, fac2))

        # Environmental impact and traffic costs
        env_impact_cost = np.random.uniform(1, 10, self.n_facilities).tolist()
        traffic_cost = np.random.uniform(1, 5, self.n_facilities).tolist()
        
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
            "machine_maintenance_schedules": machine_maintenance_schedules,
            "machine_breakdown_probabilities": machine_breakdown_probabilities,
            "employee_shift_costs": employee_shift_costs,
            "employee_shift_availability": employee_shift_availability,
            "raw_material_delivery_schedule": raw_material_delivery_schedule,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "env_impact_cost": env_impact_cost,
            "traffic_cost": traffic_cost,
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
        machine_maintenance_schedules = instance['machine_maintenance_schedules']
        machine_breakdown_probabilities = instance['machine_breakdown_probabilities']
        employee_shift_costs = instance['employee_shift_costs']
        employee_shift_availability = instance['employee_shift_availability']
        raw_material_delivery_schedule = instance['raw_material_delivery_schedule']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        env_impact_cost = instance['env_impact_cost']
        traffic_cost = instance['traffic_cost']

        model = Model("ComplexFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])
        
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="B", name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}

        # New variables
        energy_vars = {f: model.addVar(vtype="C", name=f"Energy_{f}", lb=0) for f in range(n_facilities)}
        raw_material_vars = {n: model.addVar(vtype="C", name=f"RawMaterial_{n}", lb=0) for n in range(n_neighborhoods)}
        labor_cost_vars = {f: model.addVar(vtype="C", name=f"LaborCost_{f}", lb=0) for f in range(n_facilities)}
        environmental_impact_vars = {f: model.addVar(vtype="C", name=f"EnvironmentalImpact_{f}", lb=0) for f in range(n_facilities)}
        pricing_vars = {n: model.addVar(vtype="C", name=f"Pricing_{n}", lb=0) for n in range(n_neighborhoods)}

        machine_state_vars = {f: model.addVar(vtype="B", name=f"MachineState_{f}") for f in range(n_facilities)}
        breakdown_penalty_vars = {f: model.addVar(vtype="C", name=f"BreakdownPenalty_{f}", lb=0) for f in range(n_facilities)}
        shift_vars = {(f, s): model.addVar(vtype="B", name=f"Shift_{f}_Shift_{s}") for f in range(n_facilities) for s in range(3)}
        raw_material_stock_vars = {n: model.addVar(vtype="C", name=f"RawMaterialStock_{n}", lb=0) for n in range(n_neighborhoods)}
        
        mutual_exclusivity_vars = {i: model.addVar(vtype="B", name=f"MutualExclusivity_{i}") for i in range(len(mutual_exclusivity_pairs))}

        model.setObjective(
            quicksum(financial_rewards[n] * allocation_vars[f, n] * demand_fluctuation[n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(energy_vars[f] * energy_consumption[f] for f in range(n_facilities)) -
            quicksum(labor_cost_vars[f] * labor_cost[f] for f in range(n_facilities)) -
            quicksum(environmental_impact_vars[f] * environmental_impact[f] for f in range(n_facilities)) -
            quicksum(breakdown_penalty_vars[f] * machine_breakdown_probabilities[f] for f in range(n_facilities)) -
            quicksum(env_impact_cost[f] * environmental_impact_vars[f] for f in range(n_facilities)) -
            quicksum(traffic_cost[f] * raw_material_vars[n] for f in range(n_facilities) for n in range(n_neighborhoods)),
            "maximize"
        )

        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")
        
        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Facility_{f}_Service_{n}")
        
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= capacities[f], f"Facility_{f}_Capacity")

        for f in range(n_facilities):
            model.addCons(energy_vars[f] == quicksum(allocation_vars[f, n] * energy_consumption[f] for n in range(n_neighborhoods)), f"EnergyConsumption_{f}")

        for n in range(n_neighborhoods):
            model.addCons(raw_material_vars[n] <= raw_material_availability[n], f"RawMaterial_{n}")

        for f in range(n_facilities):
            model.addCons(labor_cost_vars[f] <= labor_cost[f], f"LaborCost_{f}")

        for f in range(n_facilities):
            model.addCons(environmental_impact_vars[f] <= environmental_impact[f], f"EnvironmentalImpact_{f}")

        for n in range(n_neighborhoods):
            model.addCons(pricing_vars[n] == financial_rewards[n] * demand_fluctuation[n], f"Pricing_{n}")

        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                model.addCons(allocation_vars[f, n] * demand_fluctuation[n] <= capacities[f], f"DemandCapacity_{f}_{n}")

        for i in range(n_neighborhoods - 1):
            n1 = ordered_neighborhoods[i]
            n2 = ordered_neighborhoods[i + 1]
            for f in range(n_facilities):
                model.addCons(allocation_vars[f, n1] + allocation_vars[f, n2] <= 1, f"SOS_Constraint_Facility_{f}_Neighborhoods_{n1}_{n2}")

        # New constraints for machine maintenance and breakdown probability
        for f in range(n_facilities):
            if machine_maintenance_schedules[f]:
                model.addCons(machine_state_vars[f] == 0, f"MachineMaintenance_{f}")

        for f in range(n_facilities):
            model.addCons(breakdown_penalty_vars[f] >= machine_breakdown_probabilities[f] * quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)), f"BreakdownPenalty_{f}")

        # New constraints for employee shifts and raw material scheduling
        for f in range(n_facilities):
            for s in range(3):
                model.addCons(shift_vars[f, s] <= employee_shift_availability[f][s], f"Shift_{f}_Availability_{s}")

        for f in range(n_facilities):
            model.addCons(labor_cost_vars[f] == quicksum(shift_vars[f, s] * employee_shift_costs[f] for s in range(3)), f"TotalShiftCost_{f}")

        for n in range(n_neighborhoods):
            model.addCons(raw_material_stock_vars[n] <= raw_material_vars[n] * raw_material_delivery_schedule[n], f"RawMaterialSchedule_{n}")

        # Mutual exclusivity constraints
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
        'n_neighborhoods': 337,
        'min_fixed_cost': 750,
        'max_fixed_cost': 2108,
        'min_transport_cost': 810,
        'max_transport_cost': 984,
        'min_capacity': 693,
        'max_capacity': 1012,
        'energy_min': 0.31,
        'energy_max': 810.0,
        'raw_material_min': 2625,
        'raw_material_max': 525,
        'labor_cost_min': 1837,
        'labor_cost_max': 2700,
        'environmental_mean': 1125,
        'environmental_std': 759,
        'demand_avg': 6,
        'demand_std': 0.8,
        'breakdown_prob_min': 0.31,
        'breakdown_prob_max': 0.17,
        'shift_cost_min': 600,
        'shift_cost_max': 1600,
        'delivery_schedule_min': 0.31,
        'delivery_schedule_max': 0.73,
        'n_exclusive_pairs': 30,
    }

    location_optimizer = ComplexFacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")