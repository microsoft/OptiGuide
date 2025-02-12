import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencySupplyDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_supply_centers > 0 and self.n_disaster_zones > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_supply_cost >= 0 and self.max_supply_cost >= self.min_supply_cost
        assert self.min_supply_capacity > 0 and self.max_supply_capacity >= self.min_supply_capacity
        assert self.min_transport_time >= 0 and self.max_transport_time >= self.min_transport_time

        supply_opening_costs = np.random.randint(self.min_supply_cost, self.max_supply_cost + 1, self.n_supply_centers)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_supply_centers, self.n_disaster_zones))
        transport_times = np.random.uniform(self.min_transport_time, self.max_transport_time, (self.n_supply_centers, self.n_disaster_zones))
        capacities = np.random.randint(self.min_supply_capacity, self.max_supply_capacity + 1, self.n_supply_centers)
        
        # Generate stochastic demand
        demand_means = np.random.randint(1, 50, self.n_disaster_zones)
        demand_stds = np.random.rand(self.n_disaster_zones) * 10  # Arbitrary standard deviation
        M = np.max(capacities)
        
        # New multi-period dynamic data
        periods = 10
        environmental_impact_factors = np.random.uniform(1.0, 1.5, self.n_supply_centers)
        
        return {
            "supply_opening_costs": supply_opening_costs,
            "transport_costs": transport_costs,
            "transport_times": transport_times,
            "capacities": capacities,
            "demand_means": demand_means,
            "demand_stds": demand_stds,
            "M": M,
            "periods": periods,
            "environmental_impact_factors": environmental_impact_factors
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        supply_opening_costs = instance['supply_opening_costs']
        transport_costs = instance['transport_costs']
        transport_times = instance['transport_times']
        capacities = instance['capacities']
        demand_means = instance['demand_means']
        demand_stds = instance['demand_stds']
        M = instance['M']
        periods = instance['periods']
        environmental_impact_factors = instance['environmental_impact_factors']
        
        model = Model("EmergencySupplyDistribution")
        n_supply_centers = len(supply_opening_costs)
        n_disaster_zones = len(demand_means)
        
        # Decision variables
        supply_open_vars = {(s, t): model.addVar(vtype="B", name=f"Supply_{s}_Period_{t}") for s in range(n_supply_centers) for t in range(periods)}
        supply_assignment_vars = {(s, z, t): model.addVar(vtype="I", lb=0, name=f"Supply_{s}_Zone_{z}_Period_{t}") for s in range(n_supply_centers) for z in range(n_disaster_zones) for t in range(periods)}
        penalty_vars = {(z, t): model.addVar(vtype="C", lb=0, name=f"Penalty_Zone_{z}_Period_{t}") for z in range(n_disaster_zones) for t in range(periods)}
        stochastic_capacity_vars = {(s, t): model.addVar(vtype="C", lb=0, name=f"Capacity_Supply_{s}_Period_{t}") for s in range(n_supply_centers) for t in range(periods)}
        
        # Objective: minimize total cost (supply center opening + transport costs + transport times + penalties for unsatisfied demand + environmental impact)
        model.setObjective(
            quicksum(supply_opening_costs[s] * supply_open_vars[s, t] for s in range(n_supply_centers) for t in range(periods)) +
            quicksum(transport_costs[s, z] * supply_assignment_vars[s, z, t] for s in range(n_supply_centers) for z in range(n_disaster_zones) for t in range(periods)) +
            quicksum(transport_times[s, z] * supply_assignment_vars[s, z, t] for s in range(n_supply_centers) for z in range(n_disaster_zones) for t in range(periods)) +
            quicksum(1000 * penalty_vars[z, t] for z in range(n_disaster_zones) for t in range(periods)) +  # High penalty for unsatisfied demand
            quicksum(environmental_impact_factors[s] * supply_assignment_vars[s, z, t] for s in range(n_supply_centers) for z in range(n_disaster_zones) for t in range(periods)), 
            "minimize"
        )
        
        # Constraints: Each disaster zone's demand must be satisfied to a robust level in each period
        for z in range(n_disaster_zones):
            for t in range(periods):
                demand_robust = demand_means[z] + 1.65 * demand_stds[z]  # 95% confidence interval
                model.addCons(quicksum(supply_assignment_vars[s, z, t] for s in range(n_supply_centers)) + penalty_vars[z, t] >= demand_robust, f"Zone_{z}_Demand_Robust_Period_{t}")
        
        # Constraints: Supply center capacity constraints per period with stochastic capacities
        for s in range(n_supply_centers):
            for t in range(periods):
                model.addCons(quicksum(supply_assignment_vars[s, z, t] for z in range(n_disaster_zones)) <= stochastic_capacity_vars[s, t], f"Stochastic_Capacity_Supply_{s}_Period_{t}")
        
        # Constraints: Supplies must be transported from open supply centers in each period (Big M Formulation)
        for s in range(n_supply_centers):
            for z in range(n_disaster_zones):
                for t in range(periods):
                    model.addCons(supply_assignment_vars[s, z, t] <= M * supply_open_vars[s, t], f"BigM_Open_Supply_{s}_For_Zone_{z}_Period_{t}")
        
        # Stochastic capacity constraints linking actual capacity to a distribution
        for s in range(n_supply_centers):
            for t in range(periods):
                stochastic_capacity_mean = np.mean(capacities)
                stochastic_capacity_std = np.std(capacities)
                model.addCons(stochastic_capacity_vars[s, t] <= stochastic_capacity_mean + 1.65 * stochastic_capacity_std, f"Stochastic_Capacity_Constraint_{s}_{t}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_supply_centers': 40,
        'n_disaster_zones': 30,
        'min_transport_cost': 1,
        'max_transport_cost': 50,
        'min_supply_cost': 40,
        'max_supply_cost': 400,
        'min_supply_capacity': 30,
        'max_supply_capacity': 500,
        'min_transport_time': 0.73,
        'max_transport_time': 10.0,
    }
    
    supply_distribution_optimizer = EmergencySupplyDistribution(parameters, seed=42)
    instance = supply_distribution_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_distribution_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")