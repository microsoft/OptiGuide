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
        assert self.min_segment_cost >= 0 and self.max_segment_cost >= self.min_segment_cost
        assert self.min_capacity_factory > 0 and self.max_capacity_factory >= self.min_capacity_factory
        assert self.min_demand >= 0 and self.max_demand >= self.min_demand

        fixed_costs = np.random.randint(self.min_cost_factory, self.max_cost_factory + 1, self.n_factories)
        
        # Generate piecewise linear cost parameters
        segment_costs = np.random.randint(self.min_segment_cost, self.max_segment_cost + 1, (self.n_factories, self.n_demand_points, self.n_segments))
        segment_breakpoints = np.sort(np.random.randint(self.min_segment_size, self.max_segment_size + 1, (self.n_factories, self.n_demand_points, self.n_segments - 1)), axis=-1)

        capacities = np.random.randint(self.min_capacity_factory, self.max_capacity_factory + 1, self.n_factories)
        demands = np.random.randint(self.min_demand, self.max_demand + 1, self.n_demand_points)

        return {
            "fixed_costs": fixed_costs,
            "segment_costs": segment_costs,
            "segment_breakpoints": segment_breakpoints,
            "capacities": capacities,
            "demands": demands,
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        segment_costs = instance['segment_costs']
        segment_breakpoints = instance['segment_breakpoints']
        capacities = instance['capacities']
        demands = instance['demands']

        model = Model("SupplyChainOptimization")
        n_factories = len(fixed_costs)
        n_demand_points = len(segment_costs[0])
        n_segments = len(segment_costs[0][0])
        
        factory_vars = {f: model.addVar(vtype="B", name=f"Factory_{f}") for f in range(n_factories)}
        transport_vars = {(f, d): model.addVar(vtype="C", name=f"Transport_{f}_Demand_{d}") for f in range(n_factories) for d in range(n_demand_points)}
        segment_vars = {(f, d, s): model.addVar(vtype="C", name=f"Segment_{f}_Demand_{d}_Segment_{s}") for f in range(n_factories) for d in range(n_demand_points) for s in range(n_segments)}
        
        # Objective function: Minimize total cost (fixed + transport via piecewise linear function)
        model.setObjective(
            quicksum(fixed_costs[f] * factory_vars[f] for f in range(n_factories)) +
            quicksum(segment_costs[f][d][s] * segment_vars[f, d, s] for f in range(n_factories) for d in range(n_demand_points) for s in range(n_segments)),
            "minimize"
        )

        # Constraints
        # Demand satisfaction
        for d in range(n_demand_points):
            model.addCons(quicksum(transport_vars[f, d] for f in range(n_factories)) == demands[d], f"Demand_Satisfaction_{d}")
        
        # Capacity limits for each factory
        for f in range(n_factories):
            model.addCons(quicksum(transport_vars[f, d] for d in range(n_demand_points)) <= capacities[f] * factory_vars[f], f"Factory_Capacity_{f}")

        # Transportation only if factory is operational
        for f in range(n_factories):
            for d in range(n_demand_points):
                model.addCons(transport_vars[f, d] <= demands[d] * factory_vars[f], f"Operational_Constraint_{f}_{d}")
        
        # Piecewise linear constraints
        for f in range(n_factories):
            for d in range(n_demand_points):
                model.addCons(quicksum(segment_vars[f, d, s] for s in range(n_segments)) == transport_vars[f, d], f"Piecewise_Sum_{f}_{d}")
                for s in range(n_segments - 1):
                    model.addCons(segment_vars[f, d, s] <= segment_breakpoints[f][d][s], f"Segment_Limit_{f}_{d}_{s}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_factories': 93,
        'n_demand_points': 250,
        'min_cost_factory': 3000,
        'max_cost_factory': 5000,
        'min_segment_cost': 3,
        'max_segment_cost': 50,
        'min_segment_size': 1000,
        'max_segment_size': 1500,
        'n_segments': 8,
        'min_capacity_factory': 1400,
        'max_capacity_factory': 3000,
        'min_demand': 15,
        'max_demand': 1600,
    }

    supply_chain_optimizer = SupplyChainOptimization(parameters, seed=42)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")