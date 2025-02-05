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

        arcs_info = {'supply_capacities': supply_capacities,
                     'demand': demand,
                     'transportation_costs': transportation_costs,
                     'fixed_costs': fixed_costs,
                     'arcs': arcs}

        return arcs_info

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        supply_capacities = instance['supply_capacities']
        demand = instance['demand']
        transportation_costs = instance['transportation_costs']
        fixed_costs = instance['fixed_costs']
        arcs = instance['arcs']

        supply_nodes = self.supply_nodes
        demand_nodes = self.demand_nodes

        model = Model("SupplyChainNetwork")

        # Variables
        flow_vars = {}
        arc_vars = {}

        for i, j in arcs:
            flow_vars[(i, j)] = model.addVar(vtype="C", name=f"flow_{i}_{j}")
        for i in range(supply_nodes):
            arc_vars[i] = model.addVar(vtype="B", name=f"activate_supply_{i}")

        # Constraints
        big_M = 200  # Arbitrary large constant; adjusted for new constraints
        for i in range(supply_nodes):
            for j in range(demand_nodes):
                model.addCons(flow_vars[(i, j)] <= supply_capacities[i] * arc_vars[i], f"supply_capacity_{i}_{j}")

            # Ensure if a supply node is activated, it supplies at least a minimum level using Big-M
            min_supply_level = 20
            model.addCons(quicksum(flow_vars[(i, j)] for j in range(demand_nodes) if (i, j) in arcs) >= min_supply_level * arc_vars[i], f"min_supply_level_{i}")

        # Demand satisfaction constraints
        for j in range(demand_nodes):
            model.addCons(quicksum(flow_vars[(i, j)] for i in range(supply_nodes) if (i, j) in arcs) >= demand[j], f"demand_{j}")

        # Objective function: minimize total cost
        transportation_cost_expr = quicksum(flow_vars[(i, j)] * transportation_costs[i, j] for (i, j) in arcs)
        fixed_cost_expr = quicksum(arc_vars[i] * fixed_costs[i] for i in range(supply_nodes))
        total_cost = transportation_cost_expr + fixed_cost_expr

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'supply_nodes': 93,
        'demand_nodes': 168,
    }

    supply_chain_problem = SupplyChainNetwork(parameters, seed=seed)
    instance = supply_chain_problem.generate_instance()
    solve_status, solve_time = supply_chain_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")