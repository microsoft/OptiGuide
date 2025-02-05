import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EmergencyMedicalSuppliesOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_zones > 0 and self.n_vehicles > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_telemedicine_cost >= 0 and self.max_telemedicine_cost >= self.min_telemedicine_cost
        assert self.min_supply_cost >= 0 and self.max_supply_cost >= self.min_supply_cost

        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, self.n_vehicles)
        telemedicine_costs = np.random.randint(self.min_telemedicine_cost, self.max_telemedicine_cost + 1, self.n_zones)
        supply_costs = np.random.randint(self.min_supply_cost, self.max_supply_cost + 1, self.n_vehicles)
        demands = np.random.normal(self.mean_demand, self.std_dev_demand, self.n_zones).astype(int)
        capacities = np.random.randint(self.min_vehicle_capacity, self.max_vehicle_capacity + 1, self.n_vehicles)

        high_risk_population = np.random.choice([0, 1], size=self.n_zones, p=[0.7, 0.3])
        mobile_medical_unit_costs = np.random.randint(500, 1000, self.n_zones)

        return {
            "transport_costs": transport_costs,
            "telemedicine_costs": telemedicine_costs,
            "supply_costs": supply_costs,
            "demands": demands,
            "capacities": capacities,
            "high_risk_population": high_risk_population,
            "mobile_medical_unit_costs": mobile_medical_unit_costs
        }

    # MILP modeling
    def solve(self, instance):
        transport_costs = instance["transport_costs"]
        telemedicine_costs = instance["telemedicine_costs"]
        supply_costs = instance["supply_costs"]
        demands = instance["demands"]
        capacities = instance["capacities"]
        high_risk_population = instance["high_risk_population"]
        mobile_medical_unit_costs = instance["mobile_medical_unit_costs"]

        model = Model("EmergencyMedicalSuppliesOptimization")
        n_vehicles = len(capacities)
        n_zones = len(demands)

        # Decision variables
        vehicle_used = {v: model.addVar(vtype="B", name=f"Vehicle_{v}") for v in range(n_vehicles)}
        supply_delivery = {(v, z): model.addVar(vtype="C", name=f"Vehicle_{v}_Zone_{z}") for v in range(n_vehicles) for z in range(n_zones)}
        telemedicine_allocated = {z: model.addVar(vtype="B", name=f"Telemedicine_Zone_{z}") for z in range(n_zones)}
        mobile_unit_allocated = {z: model.addVar(vtype="B", name=f"Mobile_Unit_Zone_{z}") for z in range(n_zones)}
        unmet_demand = {z: model.addVar(vtype="C", name=f"Unmet_Demand_Zone_{z}") for z in range(n_zones)}

        # Objective function: Minimize total cost
        model.setObjective(
            quicksum(transport_costs[v] * vehicle_used[v] for v in range(n_vehicles)) +
            quicksum(supply_costs[v] * supply_delivery[v, z] for v in range(n_vehicles) for z in range(n_zones)) +
            quicksum(telemedicine_costs[z] * telemedicine_allocated[z] for z in range(n_zones)) +
            quicksum(mobile_medical_unit_costs[z] * mobile_unit_allocated[z] for z in range(n_zones)) +
            self.penalty_unmet_demand * quicksum(unmet_demand[z] for z in range(n_zones)),
            "minimize"
        )

        # Constraints: Each zone must meet its demand
        for z in range(n_zones):
            model.addCons(
                quicksum(supply_delivery[v, z] for v in range(n_vehicles)) + unmet_demand[z] >= demands[z],
                f"Zone_{z}_Demand"
            )

        # Constraints: Each vehicle cannot exceed its capacity
        for v in range(n_vehicles):
            model.addCons(
                quicksum(supply_delivery[v, z] for z in range(n_zones)) <= capacities[v], 
                f"Vehicle_{v}_Capacity"
            )

        # Constraints: Only used vehicles can make deliveries
        for v in range(n_vehicles):
            for z in range(n_zones):
                model.addCons(
                    supply_delivery[v, z] <= capacities[v] * vehicle_used[v],
                    f"Vehicle_{v}_Deliver_{z}"
                )

        # Prioritize high-risk zones
        for z in range(n_zones):
            model.addCons(
                telemedicine_allocated[z] >= high_risk_population[z],
                f"High_Risk_Telemedicine_{z}"
            )
            model.addCons(
                mobile_unit_allocated[z] >= high_risk_population[z],
                f"High_Risk_Mobile_Unit_{z}"
            )

        # Mobile medical units can only be allocated with telemedicine support
        for z in range(n_zones):
            model.addCons(
                mobile_unit_allocated[z] <= telemedicine_allocated[z],
                f"Mobile_Unit_Telemedicine_{z}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        if model.getStatus() == "optimal":
            objective_value = model.getObjVal()
        else:
            objective_value = None

        return model.getStatus(), end_time - start_time, objective_value

if __name__ == "__main__":
    seed = 42
    parameters = {
        'n_vehicles': 90,
        'n_zones': 200,
        'min_transport_cost': 10,
        'max_transport_cost': 1000,
        'min_telemedicine_cost': 500,
        'max_telemedicine_cost': 600,
        'mean_demand': 300,
        'std_dev_demand': 100,
        'min_vehicle_capacity': 1200,
        'max_vehicle_capacity': 3000,
        'min_supply_cost': 250,
        'max_supply_cost': 250,
        'penalty_unmet_demand': 600,
    }

    emergency_optimizer = EmergencyMedicalSuppliesOptimization(parameters, seed)
    instance = emergency_optimizer.generate_instance()
    solve_status, solve_time, objective_value = emergency_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")