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

    ################# Data Generation #################
    def generate_instance(self):
        assert self.n_markets > 0 and self.n_zones > 0
        assert self.min_market_cost >= 0 and self.max_market_cost >= self.min_market_cost
        assert self.min_zone_cost >= 0 and self.max_zone_cost >= self.min_zone_cost
        assert self.min_market_cap > 0 and self.max_market_cap >= self.min_market_cap

        market_costs = np.random.randint(self.min_market_cost, self.max_market_cost + 1, self.n_markets)
        zone_costs = np.random.randint(self.min_zone_cost, self.max_zone_cost + 1, (self.n_markets, self.n_zones))
        capacities = np.random.randint(self.min_market_cap, self.max_market_cap + 1, self.n_markets)
        demands = np.random.randint(1, 10, self.n_zones)

        return {
            "market_costs": market_costs,
            "zone_costs": zone_costs,
            "capacities": capacities,
            "demands": demands,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        market_costs = instance['market_costs']
        zone_costs = instance['zone_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        
        model = Model("ResidentialZoneDevelopment")
        n_markets = len(market_costs)
        n_zones = len(zone_costs[0])
        
        # Decision variables
        market_vars = {m: model.addVar(vtype="B", name=f"Market_{m}") for m in range(n_markets)}
        zone_vars = {(m, z): model.addVar(vtype="B", name=f"Market_{m}_Zone_{z}") for m in range(n_markets) for z in range(n_zones)}
        
        # Objective: minimize the total cost including market costs and zone costs
        model.setObjective(
            quicksum(market_costs[m] * market_vars[m] for m in range(n_markets)) +
            quicksum(zone_costs[m, z] * zone_vars[m, z] for m in range(n_markets) for z in range(n_zones)), 
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
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_markets': 100,
        'n_zones': 52,
        'min_zone_cost': 360,
        'max_zone_cost': 900,
        'min_market_cost': 1500,
        'max_market_cost': 1500,
        'min_market_cap': 1400,
        'max_market_cap': 3000,
    }

    zone_development_optimizer = ResidentialZoneDevelopment(parameters, seed=seed)
    instance = zone_development_optimizer.generate_instance()
    solve_status, solve_time, objective_value = zone_development_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")