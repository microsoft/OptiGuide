import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EnhancedWarehouseLayoutOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_units > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_facility_space > 0 and self.max_facility_space >= self.min_facility_space
        
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_units))
        spaces = np.random.randint(self.min_facility_space, self.max_facility_space + 1, self.n_facilities)
        demands = np.random.randint(1, 10, self.n_units)
        opening_times = np.random.randint(self.min_opening_time, self.max_opening_time + 1, self.n_facilities)
        maintenance_periods = np.random.randint(self.min_maintenance_period, self.max_maintenance_period + 1, self.n_facilities)

        energy_consumption = np.random.uniform(1.0, 5.0, (self.n_facilities, self.n_units))
        carbon_cost_coefficients = np.random.uniform(0.5, 2.0, (self.n_facilities, self.n_units))
        
        employee_shift_costs = np.random.uniform(20, 80, self.n_facilities)
        employee_shift_availability = [np.random.choice([True, False], 3, p=[0.7, 0.3]).tolist() for _ in range(self.n_facilities)]
        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            fac1 = random.randint(0, self.n_facilities - 1)
            fac2 = random.randint(0, self.n_facilities - 1)
            if fac1 != fac2:
                mutual_exclusivity_pairs.append((fac1, fac2))
                
        raw_material_delivery_schedule = np.random.uniform(0.7, 1.3, self.n_units)

        return {
            "facility_costs": facility_costs,
            "transport_costs": transport_costs,
            "spaces": spaces,
            "demands": demands,
            "opening_times": opening_times,
            "maintenance_periods": maintenance_periods,
            "energy_consumption": energy_consumption,
            "carbon_cost_coefficients": carbon_cost_coefficients,
            "employee_shift_costs": employee_shift_costs,
            "employee_shift_availability": employee_shift_availability,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "raw_material_delivery_schedule": raw_material_delivery_schedule,
        }

    def solve(self, instance):
        facility_costs = instance['facility_costs']
        transport_costs = instance['transport_costs']
        spaces = instance['spaces']
        demands = instance['demands']
        opening_times = instance['opening_times']
        maintenance_periods = instance['maintenance_periods']
        energy_consumption = instance['energy_consumption']
        carbon_cost_coefficients = instance['carbon_cost_coefficients']
        employee_shift_costs = instance['employee_shift_costs']
        employee_shift_availability = instance['employee_shift_availability']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        raw_material_delivery_schedule = instance['raw_material_delivery_schedule']
        
        model = Model("EnhancedWarehouseLayoutOptimization")
        n_facilities = len(facility_costs)
        n_units = len(transport_costs[0])
        
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        opening_time_vars = {f: model.addVar(vtype="I", lb=0, name=f"OpeningTime_{f}") for f in range(n_facilities)}
        maintenance_period_vars = {f: model.addVar(vtype="I", lb=0, name=f"MaintenancePeriod_{f}") for f in range(n_facilities)}
        
        energy_vars = {(f, u): model.addVar(vtype="C", name=f"Energy_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        carbon_emission_vars = {(f, u): model.addVar(vtype="C", name=f"Carbon_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        shift_vars = {(f, s): model.addVar(vtype="B", name=f"Shift_{f}_Shift_{s}") for f in range(n_facilities) for s in range(3)}
        raw_material_stock_vars = {u: model.addVar(vtype="C", name=f"RawMaterialStock_{u}", lb=0) for u in range(n_units)}
        mutual_exclusivity_vars = {(f1, f2): model.addVar(vtype="B", name=f"MutualExclusivity_{f1}_{f2}") for f1, f2 in mutual_exclusivity_pairs}

        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f, u] * transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)) +
            quicksum(opening_times[f] * opening_time_vars[f] for f in range(n_facilities)) +
            quicksum(maintenance_periods[f] * maintenance_period_vars[f] for f in range(n_facilities)) +
            quicksum(carbon_cost_coefficients[f, u] * carbon_emission_vars[f, u] for f in range(n_facilities) for u in range(n_units)) +
            quicksum(employee_shift_costs[f] * shift_vars[f, s] for f in range(n_facilities) for s in range(3)) -
            50 * quicksum(transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)) 
            , "minimize"
        )
        
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) == 1, f"Unit_{u}_Demand")
        
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f], f"Facility_{f}_Serve_{u}")
        
        for f in range(n_facilities):
            model.addCons(quicksum(demands[u] * transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")

        for f in range(n_facilities):
            model.addCons(opening_time_vars[f] == opening_times[f], f"Facility_{f}_OpeningTime")
            model.addCons(maintenance_period_vars[f] == maintenance_periods[f], f"Facility_{f}_MaintenancePeriod")

        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(energy_vars[f, u] == energy_consumption[f, u] * transport_vars[f, u], f"Energy_{f}_Unit_{u}")

        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(carbon_emission_vars[f, u] == carbon_cost_coefficients[f, u] * energy_vars[f, u], f"Carbon_{f}_Unit_{u}")

        model.addCons(
            quicksum(carbon_emission_vars[f, u] for f in range(n_facilities) for u in range(n_units)) <= self.sustainability_budget,
            "Total_Carbon_Emissions_Limit"
        )

        # New constraints for employee shifts and raw material scheduling
        for f in range(n_facilities):
            for s in range(3):
                model.addCons(shift_vars[f, s] <= employee_shift_availability[f][s], f"Shift_{f}_Availability_{s}")

        for u in range(n_units):
            model.addCons(raw_material_stock_vars[u] <= demands[u] * raw_material_delivery_schedule[u], f"RawMaterialSchedule_{u}")

        # Mutual exclusivity constraints
        for f1, f2 in mutual_exclusivity_pairs:
            model.addCons(facility_vars[f1] + facility_vars[f2] <= 1, f"MutualExclusivity_{f1}_{f2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 100,
        'n_units': 84,
        'min_transport_cost': 14,
        'max_transport_cost': 1000,
        'min_facility_cost': 1500,
        'max_facility_cost': 5000,
        'min_facility_space': 1125,
        'max_facility_space': 2400,
        'min_opening_time': 1,
        'max_opening_time': 4,
        'min_maintenance_period': 450,
        'max_maintenance_period': 2700,
        'sustainability_budget': 5000,
        'n_exclusive_pairs': 30,
    }

    warehouse_layout_optimizer = EnhancedWarehouseLayoutOptimization(parameters, seed=42)
    instance = warehouse_layout_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_layout_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")