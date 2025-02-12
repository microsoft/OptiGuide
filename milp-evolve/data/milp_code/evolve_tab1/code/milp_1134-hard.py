import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainStochasticOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_factories > 0 and self.n_demand_points > 0
        assert self.min_cost_factory >= 0 and self.max_cost_factory >= self.min_cost_factory
        assert self.min_cost_transport >= 0 and self.max_cost_transport >= self.min_cost_transport
        assert self.min_capacity_factory > 0 and self.max_capacity_factory >= self.min_capacity_factory
        assert self.min_demand >= 0 and self.max_demand >= self.min_demand
        assert self.min_cost_transship >= 0 and self.max_cost_transship >= self.min_cost_transship
        assert self.n_scenarios > 1, 'There must be at least two scenarios for a stochastic model'

        fixed_costs = np.random.randint(self.min_cost_factory, self.max_cost_factory + 1, self.n_factories)
        transport_costs = np.random.randint(self.min_cost_transport, self.max_cost_transport + 1, (self.n_factories, self.n_demand_points))
        capacities = np.random.randint(self.min_capacity_factory, self.max_capacity_factory + 1, self.n_factories)
        
        # Generate multiple demand scenarios
        demands = np.random.randint(self.min_demand, self.max_demand + 1, (self.n_scenarios, self.n_periods, self.n_demand_points))
        penalty_costs = np.random.uniform(10, 50, (self.n_scenarios, self.n_periods, self.n_demand_points)).tolist()
        transship_costs = np.random.randint(self.min_cost_transship, self.max_cost_transship + 1, (self.n_factories, self.n_factories))
        holding_costs = np.random.uniform(1, 10, self.n_factories).tolist()
        backlog_costs = np.random.uniform(20, 60, (self.n_scenarios, self.n_periods, self.n_demand_points)).tolist()

        # Scenario probabilities
        scenario_probabilities = np.random.uniform(0.1, 1, self.n_scenarios)
        scenario_probabilities /= scenario_probabilities.sum()  # Normalize to sum to 1

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "demands": demands,
            "penalty_costs": penalty_costs,
            "transship_costs": transship_costs,
            "holding_costs": holding_costs,
            "backlog_costs": backlog_costs,
            "scenario_probabilities": scenario_probabilities
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        penalty_costs = instance['penalty_costs']
        transship_costs = instance['transship_costs']
        holding_costs = instance['holding_costs']
        backlog_costs = instance['backlog_costs']
        scenario_probabilities = instance['scenario_probabilities']
        
        model = Model("SupplyChainStochasticOptimization")
        n_factories = len(fixed_costs)
        n_demand_points = len(transport_costs[0])
        n_periods = demands.shape[1]
        n_scenarios = len(demands)

        # Decision variables
        factory_vars = {(f, t): model.addVar(vtype="B", name=f"Factory_{f}_Period_{t}") for f in range(n_factories) for t in range(n_periods)}
        transport_vars = {(f, d, t, s): model.addVar(vtype="C", name=f"Transport_{f}_Demand_{d}_Period_{t}_Scenario_{s}") for f in range(n_factories) for d in range(n_demand_points) for t in range(n_periods) for s in range(n_scenarios)}
        unmet_demand_vars = {(d, t, s): model.addVar(vtype="C", name=f"Unmet_Demand_{d}_Period_{t}_Scenario_{s}") for d in range(n_demand_points) for t in range(n_periods) for s in range(n_scenarios)}
        backlog_vars = {(d, t, s): model.addVar(vtype="C", name=f"Backlog_Demand_{d}_Period_{t}_Scenario_{s}") for d in range(n_demand_points) for t in range(n_periods) for s in range(n_scenarios)}
        inventory_vars = {(f, t, s): model.addVar(vtype="C", name=f"Inventory_{f}_Period_{t}_Scenario_{s}") for f in range(n_factories) for t in range(n_periods) for s in range(n_scenarios)}
        transship_vars = {(i, j, t, s): model.addVar(vtype="C", name=f"Transship_{i}_to_{j}_Period_{t}_Scenario_{s}") for i in range(n_factories) for j in range(n_factories) if i != j for t in range(n_periods) for s in range(n_scenarios)}

        # Objective function: Minimize expected total cost (fixed + transport + penalty for unmet demand + holding + backlog + transshipment)
        model.setObjective(
            quicksum(scenario_probabilities[s] * (
                quicksum(fixed_costs[f] * factory_vars[f, t] for f in range(n_factories) for t in range(n_periods)) +
                quicksum(transport_costs[f][d] * transport_vars[f, d, t, s] for f in range(n_factories) for d in range(n_demand_points) for t in range(n_periods)) +
                quicksum(penalty_costs[s][t][d] * unmet_demand_vars[d, t, s] for d in range(n_demand_points) for t in range(n_periods)) +
                quicksum(holding_costs[f] * inventory_vars[f, t, s] for f in range(n_factories) for t in range(n_periods)) +
                quicksum(backlog_costs[s][t][d] * backlog_vars[d, t, s] for d in range(n_demand_points) for t in range(n_periods)) +
                quicksum(transship_costs[i][j] * transship_vars[i, j, t, s] for i in range(n_factories) for j in range(n_factories) if i != j for t in range(n_periods))
            ) for s in range(n_scenarios)),
            "minimize"
        )

        # Constraints

        # Demand satisfaction for each scenario (supplies, unmet demand, and backlog must cover total demand each period)
        for d in range(n_demand_points):
            for t in range(n_periods):
                for s in range(n_scenarios):
                    model.addCons(
                        quicksum(transport_vars[f, d, t, s] for f in range(n_factories)) + 
                        unmet_demand_vars[d, t, s] + backlog_vars[d, t, s] == demands[s][t][d], 
                        f"Demand_Satisfaction_{d}_Period_{t}_Scenario_{s}"
                    )

        # Capacity limits for each factory each period in each scenario
        for f in range(n_factories):
            for t in range(n_periods):
                for s in range(n_scenarios):
                    model.addCons(
                        quicksum(transport_vars[f, d, t, s] for d in range(n_demand_points)) + 
                        quicksum(transship_vars[f, j, t, s] for j in range(n_factories) if j != f) <= 
                        capacities[f] * factory_vars[f, t], 
                        f"Factory_Capacity_{f}_Period_{t}_Scenario_{s}"
                    )
        
        # Transportation only if factory is operational each period in each scenario
        for f in range(n_factories):
            for d in range(n_demand_points):
                for t in range(n_periods):
                    for s in range(n_scenarios):
                        model.addCons(
                            transport_vars[f, d, t, s] <= demands[s][t][d] * factory_vars[f, t], 
                            f"Operational_Constraint_{f}_{d}_Period_{t}_Scenario_{s}"
                        )
        
        # Inventory balance constraints each period in each scenario
        for f in range(n_factories):
            for t in range(1, n_periods):
                for s in range(n_scenarios):
                    model.addCons(
                        inventory_vars[f, t, s] == 
                        inventory_vars[f, t-1, s] + 
                        quicksum(transport_vars[f, d, t-1, s] for d in range(n_demand_points)) - 
                        quicksum(transport_vars[f, d, t, s] for d in range(n_demand_points)) + 
                        quicksum(transship_vars[i, f, t-1, s] for i in range(n_factories) if i != f) - 
                        quicksum(transship_vars[f, j, t, s] for j in range(n_factories) if j != f), 
                        f"Inventory_Balance_{f}_Period_{t}_Scenario_{s}"
                    )
        
        # Initial inventory is zero in all scenarios
        for f in range(n_factories):
            for s in range(n_scenarios):
                model.addCons(inventory_vars[f, 0, s] == 0, f"Initial_Inventory_{f}_Scenario_{s}")
        
        # Unmet demand and backlog balance each period in each scenario
        for d in range(n_demand_points):
            for t in range(1, n_periods):
                for s in range(n_scenarios):
                    model.addCons(
                        backlog_vars[d, t, s] == 
                        unmet_demand_vars[d, t-1, s], 
                        f"Backlog_Balance_{d}_Period_{t}_Scenario_{s}"
                    )
        
        # Symmetry breaking constraints
        for f in range(n_factories - 1):
            for t in range(n_periods):
                if fixed_costs[f] == fixed_costs[f + 1]:
                    for s in range(n_scenarios):
                        model.addCons(
                            factory_vars[f, t] >= factory_vars[f + 1, t],
                            f"Symmetry_Break_{f}_{f+1}_Period_{t}_Scenario_{s}"
                        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_factories': 5,
        'n_demand_points': 1,
        'n_periods': 450,
        'min_cost_factory': 3000,
        'max_cost_factory': 5000,
        'min_cost_transport': 0,
        'max_cost_transport': 35,
        'min_cost_transship': 18,
        'max_cost_transship': 750,
        'min_capacity_factory': 175,
        'max_capacity_factory': 187,
        'min_demand': 0,
        'max_demand': 120,
        'n_scenarios': 2,
    }
    
    supply_chain_optimizer = SupplyChainStochasticOptimization(parameters, seed)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")