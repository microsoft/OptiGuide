import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        supply_nodes = self.supply_nodes
        demand_nodes = self.demand_nodes
        arcs = [(i, j) for i in range(supply_nodes) for j in range(demand_nodes)]
        
        supply_capacities = np.random.randint(50, 150, size=supply_nodes)
        demand = np.random.randint(30, 100, size=demand_nodes)
        transportation_costs = np.random.randint(1, 20, size=(supply_nodes, demand_nodes))
        fixed_costs = np.random.randint(100, 500, size=supply_nodes)

        min_supply_levels = np.random.randint(20, 50, size=supply_nodes)  # Minimum supply levels
        surplus_penalty = np.random.randint(5, 15)  # Penalty per unit surplus supply
        seasonal_variation = np.random.normal(0, 0.1, size=supply_nodes)  # New variability parameter

        arcs_info = {'supply_capacities': supply_capacities,
                     'demand': demand,
                     'transportation_costs': transportation_costs,
                     'fixed_costs': fixed_costs,
                     'arcs': arcs,
                     'min_supply_levels': min_supply_levels,  # Include new data
                     'surplus_penalty': surplus_penalty,     # Include new data
                     'seasonal_variation': seasonal_variation}  # Include new data

        return arcs_info

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        supply_capacities = instance['supply_capacities']
        demand = instance['demand']
        transportation_costs = instance['transportation_costs']
        fixed_costs = instance['fixed_costs']
        arcs = instance['arcs']
        min_supply_levels = instance['min_supply_levels']
        surplus_penalty = instance['surplus_penalty']
        seasonal_variation = instance['seasonal_variation']

        supply_nodes = self.supply_nodes
        demand_nodes = self.demand_nodes

        model = Model("SupplyChainNetwork")

        # Variables
        flow_vars = {}
        arc_vars = {}
        surplus_vars = {}  # New surplus supply variables

        for i, j in arcs:
            flow_vars[(i, j)] = model.addVar(vtype="C", name=f"flow_{i}_{j}")
        for i in range(supply_nodes):
            arc_vars[i] = model.addVar(vtype="B", name=f"activate_supply_{i}", obj=fixed_costs[i])
            surplus_vars[i] = model.addVar(vtype="C", name=f"surplus_{i}", obj=surplus_penalty)

        # Constraints
        big_M = 1000  # Arbitrary large constant; adjust based on problem scale
        for i in range(supply_nodes):
            slope = seasonal_variation[i]
            for j in range(demand_nodes):
                model.addCons(flow_vars[(i, j)] <= (supply_capacities[i] * (1 + slope)) * arc_vars[i], f"supply_capacity_{i}_{j}")
        
            # Minimum supply level if activated (Big M Formulation)
            model.addCons(quicksum(flow_vars[(i, j)] for j in range(demand_nodes) if (i, j) in arcs) >= min_supply_levels[i] * arc_vars[i], f"min_supply_level_{i}_big_M")

        # Demand satisfaction constraints
        for j in range(demand_nodes):
            model.addCons(quicksum(flow_vars[(i, j)] for i in range(supply_nodes) if (i, j) in arcs) >= demand[j], f"demand_{j}")

        # Objective function: minimize total cost
        transportation_cost_expr = quicksum(flow_vars[(i, j)] * transportation_costs[i, j] for (i, j) in arcs)
        fixed_cost_expr = quicksum(arc_vars[i] * fixed_costs[i] for i in range(supply_nodes))
        surplus_penalty_expr = quicksum(surplus_vars[i] * surplus_penalty for i in range(supply_nodes))
        total_cost = transportation_cost_expr + fixed_cost_expr + surplus_penalty_expr
        
        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'supply_nodes': 125,
        'demand_nodes': 168,
    }

    supply_chain_problem = SupplyChainNetwork(parameters, seed=seed)
    instance = supply_chain_problem.generate_instance()
    solve_status, solve_time = supply_chain_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")