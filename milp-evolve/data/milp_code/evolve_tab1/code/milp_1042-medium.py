import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ManufacturingDistributionNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_plants > 0 and self.n_zones >= self.n_plants
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_logistics_cost >= 0 and self.max_logistics_cost >= self.min_logistics_cost
        assert self.min_production_cost >= 0 and self.max_production_cost >= self.min_production_cost
        assert self.min_hazard_exposure >= 0 and self.max_hazard_exposure >= self.min_hazard_exposure

        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_plants)
        logistics_costs = np.random.randint(self.min_logistics_cost, self.max_logistics_cost + 1, (self.n_plants, self.n_zones))
        production_costs = np.random.randint(self.min_production_cost, self.max_production_cost + 1, self.n_plants)
        demand = np.random.randint(self.min_demand, self.max_demand + 1, self.n_zones)

        hazard_exposures = np.random.randint(self.min_hazard_exposure, self.max_hazard_exposure + 1, self.n_plants)
        environmental_impact = np.random.uniform(10, 50, self.n_plants)
        
        scheduled_maintenance = np.random.choice([True, False], self.n_plants, p=[0.2, 0.8])
        breakdown_probabilities = np.random.uniform(0.01, 0.05, self.n_plants)
        
        shift_cost = np.random.uniform(20, 80, self.n_plants)
        shift_availability = [np.random.choice([True, False], 3, p=[0.7, 0.3]).tolist() for _ in range(self.n_plants)]

        raw_material_schedule = np.random.uniform(0.5, 1.5, self.n_zones)
        
        mutual_exclusivity_pairs = []
        for _ in range(self.mutual_exclusivity_count):
            p1 = random.randint(0, self.n_plants - 1)
            p2 = random.randint(0, self.n_plants - 1)
            if p1 != p2:
                mutual_exclusivity_pairs.append((p1, p2))

        # Energy consumption data generation
        energy_costs = np.random.uniform(100, 500, self.n_plants)
        energy_consumption_limits = np.random.uniform(1000, 2000, self.n_plants)

        # Labor availability and cost data generation
        labor_costs = np.random.uniform(15, 60, self.n_plants)
        worker_availability = np.random.randint(5, 20, self.n_plants)

        # Demand variability over multiple periods
        demand_variability = np.random.normal(1.0, 0.1, self.n_zones)

        return {
            "fixed_costs": fixed_costs,
            "logistics_costs": logistics_costs,
            "production_costs": production_costs,
            "demand": demand,
            "hazard_exposures": hazard_exposures,
            "environmental_impact": environmental_impact,
            "scheduled_maintenance": scheduled_maintenance,
            "breakdown_probabilities": breakdown_probabilities,
            "shift_cost": shift_cost,
            "shift_availability": shift_availability,
            "raw_material_schedule": raw_material_schedule,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "energy_costs": energy_costs,
            "energy_consumption_limits": energy_consumption_limits,
            "labor_costs": labor_costs,
            "worker_availability": worker_availability,
            "demand_variability": demand_variability
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        logistics_costs = instance['logistics_costs']
        production_costs = instance['production_costs']
        demand = instance['demand']
        hazard_exposures = instance['hazard_exposures']
        environmental_impact = instance['environmental_impact']
        scheduled_maintenance = instance['scheduled_maintenance']
        breakdown_probabilities = instance['breakdown_probabilities']
        shift_cost = instance['shift_cost']
        shift_availability = instance['shift_availability']
        raw_material_schedule = instance['raw_material_schedule']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        energy_costs = instance['energy_costs']
        energy_consumption_limits = instance['energy_consumption_limits']
        labor_costs = instance['labor_costs']
        worker_availability = instance['worker_availability']
        demand_variability = instance['demand_variability']

        model = Model("ManufacturingDistributionNetwork")
        n_plants = len(fixed_costs)
        n_zones = len(logistics_costs[0])

        plant_vars = {p: model.addVar(vtype="B", name=f"Plant_{p}") for p in range(n_plants)}
        distribution_vars = {(p, z): model.addVar(vtype="B", name=f"Plant_{p}_Zone_{z}") for p in range(n_plants) for z in range(n_zones)}

        hazard_vars = {p: model.addVar(vtype="C", name=f"Hazard_{p}", lb=0) for p in range(n_plants)}
        production_vars = {p: model.addVar(vtype="C", name=f"Production_{p}", lb=0) for p in range(n_plants)}
        breakdown_vars = {p: model.addVar(vtype="C", name=f"Breakdown_{p}", lb=0) for p in range(n_plants)}
        environmental_vars = {p: model.addVar(vtype="C", name=f"Environment_{p}", lb=0) for p in range(n_plants)}
        shift_vars = {(p, s): model.addVar(vtype="B", name=f"Shift_{p}_{s}") for p in range(n_plants) for s in range(3)}
        raw_material_vars = {z: model.addVar(vtype="C", name=f"RawMaterial_{z}", lb=0) for z in range(n_zones)}

        # New Variables for energy and labor
        energy_vars = {p: model.addVar(vtype="C", name=f"Energy_{p}", lb=0) for p in range(n_plants)}
        labor_vars = {p: model.addVar(vtype="C", name=f"Labor_{p}", lb=0) for p in range(n_plants)}

        # New Variable for Convex Hull Formulation
        convex_hull_vars = {(p1, p2): model.addVar(vtype="B", name=f"ConvexHull_{p1}_{p2}") for (p1, p2) in mutual_exclusivity_pairs}

        # Objective
        model.setObjective(
            quicksum(demand[z] * demand_variability[z] * distribution_vars[p, z] for p in range(n_plants) for z in range(n_zones)) -
            quicksum(fixed_costs[p] * plant_vars[p] for p in range(n_plants)) -
            quicksum(logistics_costs[p][z] * distribution_vars[p, z] for p in range(n_plants) for z in range(n_zones)) -
            quicksum(hazard_exposures[p] * hazard_vars[p] for p in range(n_plants)) -
            quicksum(production_costs[p] * production_vars[p] for p in range(n_plants)) -
            quicksum(environmental_vars[p] * environmental_impact[p] for p in range(n_plants)) -
            quicksum(breakdown_vars[p] * breakdown_probabilities[p] for p in range(n_plants)) -
            quicksum(energy_vars[p] * energy_costs[p] for p in range(n_plants)) -
            quicksum(labor_vars[p] * labor_costs[p] for p in range(n_plants)),
            "maximize"
        )

        # Existing Constraints
        for z in range(n_zones):
            model.addCons(quicksum(distribution_vars[p, z] for p in range(n_plants)) == 1, f"Zone_{z}_Assignment")

        for p in range(n_plants):
            for z in range(n_zones):
                model.addCons(distribution_vars[p, z] <= plant_vars[p], f"Plant_{p}_Service_{z}")

        for p in range(n_plants):
            model.addCons(production_vars[p] <= production_costs[p], f"Production_{p}")

        for p in range(n_plants):
            model.addCons(hazard_vars[p] <= hazard_exposures[p], f"HazardExposure_{p}")

        for p in range(n_plants):
            model.addCons(environmental_vars[p] <= environmental_impact[p], f"EnvironmentalImpact_{p}")

        for p in range(n_plants):
            if scheduled_maintenance[p]:
                model.addCons(plant_vars[p] == 0, f"ScheduledMaintenance_{p}")

        for p in range(n_plants):
            model.addCons(breakdown_vars[p] >= breakdown_probabilities[p] * quicksum(distribution_vars[p, z] for z in range(n_zones)), f"BreakdownRisk_{p}")

        for p in range(n_plants):
            for s in range(3):
                model.addCons(shift_vars[p, s] <= shift_availability[p][s], f"ShiftAvailability_{p}_{s}")

        for z in range(n_zones):
            model.addCons(raw_material_vars[z] <= raw_material_schedule[z], f"RawMaterialAvailability_{z}")

        for i, (p1, p2) in enumerate(mutual_exclusivity_pairs):
            model.addCons(plant_vars[p1] + plant_vars[p2] <= 1, f"MutualExclusivity_{p1}_{p2}")

        # New Constraints for energy and labor
        for p in range(n_plants):
            model.addCons(energy_vars[p] <= energy_consumption_limits[p], f"EnergyLimit_{p}")
        
        for p in range(n_plants):
            model.addCons(labor_vars[p] <= worker_availability[p], f"LaborAvailability_{p}")

        # New Constraints for Convex Hull Formulation on Mutual Exclusivity
        for (p1, p2) in mutual_exclusivity_pairs:
            model.addCons(plant_vars[p1] + plant_vars[p2] <= 1 + convex_hull_vars[(p1, p2)], f"ConvexHull_Constraint_1_{p1}_{p2}")
            model.addCons(convex_hull_vars[(p1, p2)] <= plant_vars[p1], f"ConvexHull_Constraint_2_{p1}_{p2}")
            model.addCons(convex_hull_vars[(p1, p2)] <= plant_vars[p2], f"ConvexHull_Constraint_3_{p1}_{p2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_plants': 33,
        'n_zones': 300,
        'min_fixed_cost': 750,
        'max_fixed_cost': 5000,
        'min_logistics_cost': 937,
        'max_logistics_cost': 3000,
        'min_production_cost': 3000,
        'max_production_cost': 3000,
        'min_demand': 1200,
        'max_demand': 5000,
        'min_hazard_exposure': 900,
        'max_hazard_exposure': 2000,
        'environmental_mean': 675,
        'environmental_std': 3000,
        'mutual_exclusivity_count': 270,
    }
    
    network_optimizer = ManufacturingDistributionNetwork(parameters, seed=42)
    instance = network_optimizer.generate_instance()
    solve_status, solve_time, objective_value = network_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")