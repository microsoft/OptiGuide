import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class OptimalProductionScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)

    def extra_hours_costs(self):
        return np.random.uniform(20, 40, self.n_products)

    def equipment_costs(self):
        return np.random.uniform(100, 500, self.n_equipment)

    def raw_material_costs(self):
        return np.random.uniform(5, 15, self.n_materials)

    def generate_instance(self):
        demand = self.randint(self.n_products, self.demand_interval)
        equipment_capacities = self.randint(self.n_equipment, self.equipment_capacity_interval)
        material_availabilities = self.randint(self.n_materials, self.material_availability_interval)
        production_times = np.random.uniform(1, 4, (self.n_products, self.n_equipment))
        material_requirements = np.random.uniform(0.5, 2.5, (self.n_products, self.n_materials))

        overtime_costs = self.extra_hours_costs()
        equipment_costs = self.equipment_costs()
        raw_material_costs = self.raw_material_costs()

        res = {
            'demand': demand,
            'equipment_capacities': equipment_capacities,
            'material_availabilities': material_availabilities,
            'production_times': production_times,
            'material_requirements': material_requirements,
            'overtime_costs': overtime_costs,
            'equipment_costs': equipment_costs,
            'raw_material_costs': raw_material_costs,
        }

        return res

    def solve(self, instance):
        demand = instance['demand']
        equipment_capacities = instance['equipment_capacities']
        material_availabilities = instance['material_availabilities']
        production_times = instance['production_times']
        material_requirements = instance['material_requirements']
        overtime_costs = instance['overtime_costs']
        equipment_costs = instance['equipment_costs']
        raw_material_costs = instance['raw_material_costs']

        n_products = len(demand)
        n_equipment = len(equipment_capacities)
        n_materials = len(material_availabilities)

        model = Model("OptimalProductionScheduling")

        # Decision variables
        produce = {(i, j): model.addVar(vtype="C", name=f"Produce_Product_{i}_Equipment_{j}") for i in range(n_products) for j in range(n_equipment)}
        use_overtime = {i: model.addVar(vtype="C", name=f"Use_Overtime_{i}") for i in range(n_products)}
        use_equipment = {j: model.addVar(vtype="B", name=f"Use_Equipment_{j}") for j in range(n_equipment)}
        use_material = {(i, k): model.addVar(vtype="C", name=f"Use_Material_{i}_Material_{k}") for i in range(n_products) for k in range(n_materials)}

        # Objective: Minimize total costs (overtime, equipment, and raw materials)
        overtime_weight = 1.0
        equipment_weight = 1.0
        material_weight = 1.0

        objective_expr = quicksum(overtime_weight * overtime_costs[i] * use_overtime[i] for i in range(n_products)) + \
                         quicksum(equipment_weight * equipment_costs[j] * use_equipment[j] for j in range(n_equipment)) + \
                         quicksum(material_weight * raw_material_costs[k] * use_material[i, k] for i in range(n_products) for k in range(n_materials))

        # Constraints: Meet production demands
        for i in range(n_products):
            model.addCons(quicksum(produce[i, j] for j in range(n_equipment)) + use_overtime[i] >= demand[i], f"Demand_{i}")

        # Constraints: Equipment capacity limits
        for j in range(n_equipment):
            model.addCons(quicksum(produce[i, j] * production_times[i, j] for i in range(n_products)) <= equipment_capacities[j] * use_equipment[j], f"Equipment_Capacity_{j}")

        # Constraints: Material availability limits
        for k in range(n_materials):
            model.addCons(quicksum(use_material[i, k] for i in range(n_products)) <= material_availabilities[k], f"Material_Availability_{k}")

        # Constraints: Each product's material usage limits
        for i in range(n_products):
            for k in range(n_materials):
                model.addCons(use_material[i, k] <= material_requirements[i, k] * quicksum(produce[i, j] for j in range(n_equipment)), f"Material_Usage_{i}_{k}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_products': 300,
        'n_equipment': 140,
        'n_materials': 75,
        'demand_interval': (250, 750),
        'equipment_capacity_interval': (1000, 2000),
        'material_availability_interval': (900, 1800),
    }
    
    optimal_production_scheduling = OptimalProductionScheduling(parameters, seed=seed)
    instance = optimal_production_scheduling.generate_instance()
    solve_status, solve_time = optimal_production_scheduling.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")