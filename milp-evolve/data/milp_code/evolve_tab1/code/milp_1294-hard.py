import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ResidentialZoneDevelopment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_markets > 0 and self.n_zones > 0
        assert self.min_market_cost >= 0 and self.max_market_cost >= self.min_market_cost
        assert self.min_zone_cost >= 0 and self.max_zone_cost >= self.min_zone_cost
        assert self.min_market_cap > 0 and self.max_market_cap >= self.min_market_cap

        market_costs = np.random.randint(self.min_market_cost, self.max_market_cost + 1, self.n_markets)
        zone_costs = np.random.randint(self.min_zone_cost, self.max_zone_cost + 1, (self.n_markets, self.n_zones))
        capacities = np.random.randint(self.min_market_cap, self.max_market_cap + 1, self.n_markets)
        demands = np.random.randint(1, 10, self.n_zones)
        environmental_impact = {z: np.random.uniform(0, 1) for z in range(self.n_zones)}
        maintenance_costs = np.random.uniform(self.min_maintenance_cost, self.max_maintenance_cost, self.n_markets)

        mutual_exclusivity_pairs = []
        for _ in range(self.n_exclusive_pairs):
            m1 = random.randint(0, self.n_markets - 1)
            m2 = random.randint(0, self.n_markets - 1)
            if m1 != m2:
                mutual_exclusivity_pairs.append((m1, m2))

        # New Data for Enhanced Complexity
        renewable_energy_costs = np.random.uniform(5, 15, (self.n_markets, self.n_zones))
        carbon_emission_limits = np.random.uniform(50, 100, self.n_zones)
        inventory_holding_costs = np.random.uniform(3, 7, self.n_zones)

        return {
            "market_costs": market_costs,
            "zone_costs": zone_costs,
            "capacities": capacities,
            "demands": demands,
            "environmental_impact": environmental_impact,
            "maintenance_costs": maintenance_costs,
            "mutual_exclusivity_pairs": mutual_exclusivity_pairs,
            "renewable_energy_costs": renewable_energy_costs,
            "carbon_emission_limits": carbon_emission_limits,
            "inventory_holding_costs": inventory_holding_costs
        }

    def solve(self, instance):
        market_costs = instance['market_costs']
        zone_costs = instance['zone_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        environmental_impact = instance['environmental_impact']
        maintenance_costs = instance['maintenance_costs']
        mutual_exclusivity_pairs = instance["mutual_exclusivity_pairs"]
        
        renewable_energy_costs = instance["renewable_energy_costs"]
        carbon_emission_limits = instance["carbon_emission_limits"]
        inventory_holding_costs = instance["inventory_holding_costs"]

        model = Model("ResidentialZoneDevelopment")
        n_markets = len(market_costs)
        n_zones = len(zone_costs[0])

        # Decision variables
        market_vars = {m: model.addVar(vtype="B", name=f"Market_{m}") for m in range(n_markets)}
        zone_vars = {(m, z): model.addVar(vtype="B", name=f"Market_{m}_Zone_{z}") for m in range(n_markets) for z in range(n_zones)}
        maintenance_vars = {m: model.addVar(vtype="C", name=f"Maintenance_{m}") for m in range(n_markets)}
        environmental_impact_vars = {z: model.addVar(vtype="C", name=f"EnvImpact_{z}") for z in range(n_zones)}

        # New variables for added complexity
        renewable_energy_vars = {(m, z): model.addVar(vtype="B", name=f"Renewable_{m}_{z}") for m in range(n_markets) for z in range(n_zones)}
        carbon_emission_vars = {z: model.addVar(vtype="C", name=f"Carbon_{z}") for z in range(n_zones)}
        inventory_vars = {z: model.addVar(vtype="C", name=f"Inventory_{z}") for z in range(n_zones)}

        # Objective: minimize the total cost including market costs, zone costs, maintenance costs, and environmental impacts
        model.setObjective(
            quicksum(market_costs[m] * market_vars[m] for m in range(n_markets)) +
            quicksum(zone_costs[m, z] * zone_vars[m, z] for m in range(n_markets) for z in range(n_zones)) + 
            quicksum(maintenance_vars[m] for m in range(n_markets)) +
            quicksum(environmental_impact_vars[z] for z in range(n_zones)) +
            quicksum(renewable_energy_costs[m, z] * renewable_energy_vars[m, z] for m in range(n_markets) for z in range(n_zones)) +
            quicksum(inventory_holding_costs[z] * inventory_vars[z] for z in range(n_zones)),
            "minimize"
        )

        # Constraints: Each zone demand is met by exactly one market
        for z in range(n_zones):
            model.addCons(quicksum(zone_vars[m, z] for m in range(n_markets)) == 1, f"Zone_{z}_Demand")
        
        # Constraints: Only open markets can serve zones
        for m in range(n_markets):
            for z in range(n_zones):
                model.addCons(zone_vars[m, z] <= market_vars[m], f"Market_{m}_Serve_{z}")
        
        # Constraints: Markets cannot exceed their capacities
        for m in range(n_markets):
            model.addCons(quicksum(demands[z] * zone_vars[m, z] for z in range(n_zones)) <= capacities[m], f"Market_{m}_Capacity")
        
        # Maintenance Cost Constraints
        for m in range(n_markets):
            model.addCons(maintenance_vars[m] == self.maintenance_multiplier * market_vars[m], f"MaintenanceCost_{m}")

        # Environmental Impact Constraints
        for z in range(n_zones):
            model.addCons(environmental_impact_vars[z] == environmental_impact[z] * quicksum(zone_vars[m, z] for m in range(n_markets)), f"EnvImpact_{z}")

        # Mutual Exclusivity Constraints
        for (m1, m2) in mutual_exclusivity_pairs:
            model.addCons(market_vars[m1] + market_vars[m2] <= 1, f"Exclusive_{m1}_{m2}")

        # New renewable energy constraints
        for m in range(n_markets):
            for z in range(n_zones):
                model.addCons(renewable_energy_vars[m, z] <= zone_vars[m, z], f"RenewableEnergy_{m}_{z}")

        # Carbon Emission Limits for zones
        for z in range(n_zones):
            model.addCons(carbon_emission_vars[z] <= carbon_emission_limits[z], f"CarbonLimit_{z}")

        # Inventory Constraints
        for z in range(n_zones):
            model.addCons(inventory_vars[z] >= demands[z], f"Inventory_{z}_Demand")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_markets': 150,
        'n_zones': 78,
        'min_zone_cost': 1800,
        'max_zone_cost': 2025,
        'min_market_cost': 750,
        'max_market_cost': 3000,
        'min_market_cap': 1400,
        'max_market_cap': 3000,
        'min_maintenance_cost': 1406,
        'max_maintenance_cost': 5000,
        'maintenance_multiplier': 54.0,
        'n_exclusive_pairs': 45,
        'renewable_percentage': 0.24,
        'carbon_limit': 900.0,
        'inventory_holding_cost_mean': 45.0,
        'inventory_holding_cost_sd': 2.0,
    }

    zone_development_optimizer = ResidentialZoneDevelopment(parameters, seed=seed)
    instance = zone_development_optimizer.generate_instance()
    solve_status, solve_time, objective_value = zone_development_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")