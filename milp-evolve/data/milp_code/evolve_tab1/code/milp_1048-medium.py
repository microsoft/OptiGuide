import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainOptimization:
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

        fixed_costs = np.random.randint(self.min_cost_factory, self.max_cost_factory + 1, self.n_factories)
        transport_costs = np.random.randint(self.min_cost_transport, self.max_cost_transport + 1, (self.n_factories, self.n_demand_points))
        capacities = np.random.randint(self.min_capacity_factory, self.max_capacity_factory + 1, self.n_factories)
        demands = np.random.randint(self.min_demand, self.max_demand + 1, self.n_demand_points)

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "demands": demands,
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demands = instance['demands']

        model = Model("SupplyChainOptimization")
        n_factories = len(fixed_costs)
        n_demand_points = len(transport_costs[0])
        
        factory_vars = {f: model.addVar(vtype="B", name=f"Factory_{f}") for f in range(n_factories)}
        transport_vars = {(f, d): model.addVar(vtype="C", name=f"Transport_{f}_Demand_{d}") for f in range(n_factories) for d in range(n_demand_points)}

        # Objective function: Minimize total cost (fixed + transport)
        model.setObjective(
            quicksum(fixed_costs[f] * factory_vars[f] for f in range(n_factories)) +
            quicksum(transport_costs[f][d] * transport_vars[f, d] for f in range(n_factories) for d in range(n_demand_points)),
            "minimize"
        )

        # Constraints
        # Demand satisfaction (total supplies must cover total demand)
        for d in range(n_demand_points):
            model.addCons(quicksum(transport_vars[f, d] for f in range(n_factories)) == demands[d], f"Demand_Satisfaction_{d}")
        
        # Capacity limits for each factory
        for f in range(n_factories):
            model.addCons(quicksum(transport_vars[f, d] for d in range(n_demand_points)) <= capacities[f] * factory_vars[f], f"Factory_Capacity_{f}")

        # Transportation only if factory is operational
        for f in range(n_factories):
            for d in range(n_demand_points):
                model.addCons(transport_vars[f, d] <= demands[d] * factory_vars[f], f"Operational_Constraint_{f}_{d}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_factories': 187,
        'n_demand_points': 125,
        'min_cost_factory': 3000,
        'max_cost_factory': 5000,
        'min_cost_transport': 15,
        'max_cost_transport': 50,
        'min_capacity_factory': 140,
        'max_capacity_factory': 1000,
        'min_demand': 5,
        'max_demand': 800,
    }

    supply_chain_optimizer = SupplyChainOptimization(parameters, seed=42)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")