import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HospitalBedOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_hospitals > 0 and self.n_zones > 0
        assert self.min_hospital_cost >= 0 and self.max_hospital_cost >= self.min_hospital_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_bed_capacity > 0 and self.max_bed_capacity >= self.min_bed_capacity

        hospital_costs = np.random.randint(self.min_hospital_cost, self.max_hospital_cost + 1, self.n_hospitals)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_hospitals, self.n_zones))
        bed_capacities = np.random.randint(self.min_bed_capacity, self.max_bed_capacity + 1, self.n_hospitals)
        patient_needs = np.random.randint(1, 10, self.n_zones)
        equipment_limits = np.random.uniform(self.min_equipment_limit, self.max_equipment_limit, self.n_hospitals)
        distances = np.random.uniform(0, self.max_distance, (self.n_hospitals, self.n_zones))
        
        # Define resource data
        resource_supply_costs = np.random.randint(self.min_resource_supply_cost, self.max_resource_supply_cost + 1, self.n_hospitals)
        resource_demands = np.random.randint(1, 5, self.n_zones)
        big_m = (bed_capacities.max() + bed_capacities.min()) * len(bed_capacities)  # Big M for resource flows

        return {
            "hospital_costs": hospital_costs,
            "transport_costs": transport_costs,
            "bed_capacities": bed_capacities,
            "patient_needs": patient_needs,
            "equipment_limits": equipment_limits,
            "distances": distances,
            "resource_supply_costs": resource_supply_costs,
            "resource_demands": resource_demands,
            "big_m": big_m,
        }

    def solve(self, instance):
        hospital_costs = instance['hospital_costs']
        transport_costs = instance['transport_costs']
        bed_capacities = instance['bed_capacities']
        patient_needs = instance['patient_needs']
        equipment_limits = instance['equipment_limits']
        distances = instance['distances']
        resource_supply_costs = instance['resource_supply_costs']
        resource_demands = instance['resource_demands']
        big_m = instance['big_m']

        model = Model("HospitalBedOptimization")
        n_hospitals = len(hospital_costs)
        n_zones = len(transport_costs[0])

        # Decision variables
        open_vars = {h: model.addVar(vtype="B", name=f"Hospital_{h}") for h in range(n_hospitals)}
        bed_vars = {(h, z): model.addVar(vtype="C", name=f"Beds_{h}_{z}") for h in range(n_hospitals) for z in range(n_zones)}

        # New Variables for resource flow
        resource_flow_vars = {(h, z): model.addVar(vtype="C", name=f"Resource_{h}_{z}") for h in range(n_hospitals) for z in range(n_zones)}
        open_resource_vars = {h: model.addVar(vtype="B", name=f"Resource_Allocation_{h}") for h in range(n_hospitals)}

        # Objective: minimize the total cost including hospital operating costs, transport costs, and resource supply costs.
        model.setObjective(
            quicksum(hospital_costs[h] * open_vars[h] for h in range(n_hospitals)) +
            quicksum(transport_costs[h, z] * bed_vars[h, z] for h in range(n_hospitals) for z in range(n_zones)) +
            quicksum(resource_supply_costs[h] * resource_flow_vars[h, z] for h in range(n_hospitals) for z in range(n_zones)),
            "minimize"
        )

        # Constraints: Each zone's patient needs are met by the hospitals
        for z in range(n_zones):
            model.addCons(quicksum(bed_vars[h, z] for h in range(n_hospitals)) == patient_needs[z], f"Zone_{z}_Patient_Needs")
        
        # Constraints: Only open hospitals can allocate beds
        for h in range(n_hospitals):
            for z in range(n_zones):
                model.addCons(bed_vars[h, z] <= equipment_limits[h] * open_vars[h], f"Hospital_{h}_Serve_{z}")
        
        # Constraints: Hospitals cannot exceed their bed capacities
        for h in range(n_hospitals):
            model.addCons(quicksum(bed_vars[h, z] for z in range(n_zones)) <= bed_capacities[h], f"Hospital_{h}_Bed_Capacity")

        # Movement distance constraint (Critical zone)
        for z in range(n_zones):
            model.addCons(quicksum(open_vars[h] for h in range(n_hospitals) if distances[h, z] <= self.max_distance) >= 1, f"Zone_{z}_Critical")

        # New Constraints for resource flows
        for z in range(n_zones):
            model.addCons(quicksum(resource_flow_vars[h, z] for h in range(n_hospitals)) == resource_demands[z], f"Zone_{z}_Resource_Demands")

        for h in range(n_hospitals):
            model.addCons(quicksum(resource_flow_vars[h, z] for z in range(n_zones)) <= open_resource_vars[h] * big_m, f"Hospital_{h}_Resource_Flow")

        model.addCons(quicksum(open_resource_vars[h] for h in range(n_hospitals)) <= n_hospitals, "Total_Hospitals_Resource_Allocation")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hospitals': 70,
        'n_zones': 360,
        'min_transport_cost': 500,
        'max_transport_cost': 2400,
        'min_hospital_cost': 1600,
        'max_hospital_cost': 1875,
        'min_bed_capacity': 375,
        'max_bed_capacity': 2000,
        'min_equipment_limit': 360,
        'max_equipment_limit': 712,
        'max_distance': 2100,
        'min_resource_supply_cost': 200,
        'max_resource_supply_cost': 700,
    }
    
    hospital_optimizer = HospitalBedOptimization(parameters, seed=seed)
    instance = hospital_optimizer.generate_instance()
    solve_status, solve_time, objective_value = hospital_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")