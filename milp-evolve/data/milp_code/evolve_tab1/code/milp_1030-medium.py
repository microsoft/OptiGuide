import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class HospitalServiceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.num_hospitals > 0 and self.num_regions >= self.num_hospitals
        assert self.min_setup_cost >= 0 and self.max_setup_cost >= self.min_setup_cost
        assert self.min_operational_cost >= 0 and self.max_operational_cost >= self.min_operational_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        setup_costs = np.random.randint(self.min_setup_cost, self.max_setup_cost + 1, self.num_hospitals)
        operational_costs = np.random.randint(self.min_operational_cost, self.max_operational_cost + 1, (self.num_hospitals, self.num_regions))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.num_hospitals)

        healthcare_demand = np.random.uniform(50, 300, self.num_regions)

        staff_availability = np.random.uniform(5, 25, self.num_hospitals).tolist()
        medical_equipment_cost = np.random.uniform(1e3, 5e3, self.num_hospitals).tolist()
        ambulance_availability = np.random.uniform(2, 10, self.num_hospitals).tolist()
        infrastructure_cost = np.random.normal(1e5, 2e4, self.num_hospitals).tolist()

        demand_variability = np.random.normal(1.0, 0.1, self.num_regions).tolist()
        permuted_regions = list(np.random.permutation(self.num_regions))
        
        return {
            "setup_costs": setup_costs,
            "operational_costs": operational_costs,
            "capacities": capacities,
            "healthcare_demand": healthcare_demand,
            "staff_availability": staff_availability,
            "medical_equipment_cost": medical_equipment_cost,
            "ambulance_availability": ambulance_availability,
            "infrastructure_cost": infrastructure_cost,
            "demand_variability": demand_variability,
            "permuted_regions": permuted_regions,
        }

    def solve(self, instance):
        setup_costs = instance['setup_costs']
        operational_costs = instance['operational_costs']
        capacities = instance['capacities']
        healthcare_demand = instance['healthcare_demand']
        staff_availability = instance['staff_availability']
        medical_equipment_cost = instance['medical_equipment_cost']
        ambulance_availability = instance['ambulance_availability']
        infrastructure_cost = instance['infrastructure_cost']
        demand_variability = instance['demand_variability']
        permuted_regions = instance['permuted_regions']

        model = Model("HospitalServiceAllocation")
        num_hospitals = len(setup_costs)
        num_regions = len(operational_costs[0])
        
        hospital_vars = {h: model.addVar(vtype="B", name=f"Hospital_{h}") for h in range(num_hospitals)}
        allocation_vars = {(h, r): model.addVar(vtype="B", name=f"Hospital_{h}_Region_{r}") for h in range(num_hospitals) for r in range(num_regions)}

        # New variables
        staff_vars = {h: model.addVar(vtype="C", name=f"Staff_{h}", lb=0) for h in range(num_hospitals)}
        equipment_vars = {h: model.addVar(vtype="C", name=f"Equipment_{h}", lb=0) for h in range(num_hospitals)}
        ambulance_vars = {h: model.addVar(vtype="C", name=f"Ambulance_{h}", lb=0) for h in range(num_hospitals)}
        infrastructure_vars = {h: model.addVar(vtype="C", name=f"Infrastructure_{h}", lb=0) for h in range(num_hospitals)}
        service_vars = {r: model.addVar(vtype="C", name=f"Service_{r}", lb=0) for r in range(num_regions)}

        model.setObjective(
            quicksum(healthcare_demand[r] * allocation_vars[h, r] * demand_variability[r] for h in range(num_hospitals) for r in range(num_regions)) -
            quicksum(setup_costs[h] * hospital_vars[h] for h in range(num_hospitals)) -
            quicksum(operational_costs[h][r] * allocation_vars[h, r] for h in range(num_hospitals) for r in range(num_regions)) -
            quicksum(staff_vars[h] * staff_availability[h] for h in range(num_hospitals)) -
            quicksum(equipment_vars[h] * medical_equipment_cost[h] for h in range(num_hospitals)) -
            quicksum(ambulance_vars[h] * ambulance_availability[h] for h in range(num_hospitals)) -
            quicksum(infrastructure_vars[h] * infrastructure_cost[h] for h in range(num_hospitals)),
            "maximize"
        )

        # Ensure at least one hospital serves each region (Set Covering Constraint)
        for r in range(num_regions):
            model.addCons(quicksum(allocation_vars[h, r] for h in range(num_hospitals)) >= 1, f"Region_{r}_Coverage")

        for h in range(num_hospitals):
            for r in range(num_regions):
                model.addCons(allocation_vars[h, r] <= hospital_vars[h], f"Hospital_{h}_Service_{r}")
        
        for h in range(num_hospitals):
            model.addCons(quicksum(allocation_vars[h, r] for r in range(num_regions)) <= capacities[h], f"Hospital_{h}_Capacity")

        for h in range(num_hospitals):
            model.addCons(staff_vars[h] == quicksum(allocation_vars[h, r] * staff_availability[h] for r in range(num_regions)), f"StaffAvailability_{h}")

        for h in range(num_hospitals):
            model.addCons(equipment_vars[h] <= medical_equipment_cost[h], f"Equipment_{h}")

        for h in range(num_hospitals):
            model.addCons(ambulance_vars[h] <= ambulance_availability[h], f"Ambulance_{h}")

        for h in range(num_hospitals):
            model.addCons(infrastructure_vars[h] <= infrastructure_cost[h], f"Infrastructure_{h}")

        for r in range(num_regions):
            model.addCons(service_vars[r] == healthcare_demand[r] * demand_variability[r], f"Service_{r}")

        for h in range(num_hospitals):
            for r in range(num_regions):
                model.addCons(allocation_vars[h, r] * demand_variability[r] <= capacities[h], f"DemandCapacity_{h}_{r}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_hospitals': 13,
        'num_regions': 1050,
        'min_setup_cost': 100000,
        'max_setup_cost': 500000,
        'min_operational_cost': 800,
        'max_operational_cost': 2000,
        'min_capacity': 50,
        'max_capacity': 1200,
    }

    hospital_optimizer = HospitalServiceAllocation(parameters, seed=42)
    instance = hospital_optimizer.generate_instance()
    solve_status, solve_time, objective_value = hospital_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")