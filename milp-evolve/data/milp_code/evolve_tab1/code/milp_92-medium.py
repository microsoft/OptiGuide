import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class AgriculturalDistributionNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        storage_nodes = self.storage_nodes
        port_nodes = self.port_nodes
        arcs = [(i, j) for i in range(storage_nodes) for j in range(port_nodes)]
        
        harvest_capacities = np.random.randint(50, 150, size=storage_nodes)
        port_demand = np.random.randint(30, 100, size=port_nodes)
        shipping_costs = np.random.randint(1, 20, size=(storage_nodes, port_nodes))
        fixed_costs = np.random.randint(100, 500, size=storage_nodes)
        
        min_harvest_levels = np.random.randint(20, 50, size=storage_nodes) 
        carbon_emission_rates = np.random.uniform(0.1, 1.0, size=(storage_nodes, port_nodes))

        arcs_info = {'harvest_capacities': harvest_capacities,
                     'port_demand': port_demand,
                     'shipping_costs': shipping_costs,
                     'fixed_costs': fixed_costs,
                     'carbon_emission_rates': carbon_emission_rates,   # New data for carbon emission rates
                     'arcs': arcs,
                     'min_harvest_levels': min_harvest_levels}          # Include new data

        return arcs_info

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        harvest_capacities = instance['harvest_capacities']
        port_demand = instance['port_demand']
        shipping_costs = instance['shipping_costs']
        fixed_costs = instance['fixed_costs']
        carbon_emission_rates = instance['carbon_emission_rates']
        arcs = instance['arcs']
        min_harvest_levels = instance['min_harvest_levels']
        
        storage_nodes = self.storage_nodes
        port_nodes = self.port_nodes

        model = Model("AgriculturalDistributionNetwork")

        # Variables
        flow_vars = {}
        arc_vars = {}
        carbon_vars = {}  # New carbon emission variables

        for i, j in arcs:
            flow_vars[(i, j)] = model.addVar(vtype="C", name=f"flow_{i}_{j}")
            arc_vars[i] = model.addVar(vtype="B", name=f"activate_storage_{i}", obj=fixed_costs[i])
            carbon_vars[(i, j)] = model.addVar(vtype="C", name=f"carbon_{i}_{j}")

        # Constraints
        # Harvest capacity constraints
        for i in range(storage_nodes):
            model.addCons(quicksum(flow_vars[(i, j)] for j in range(port_nodes) if (i, j) in arcs) <= harvest_capacities[i] * arc_vars[i], f"harvest_capacity_{i}")

            # Minimum harvest level if activated (Big M Formulation)
            M = np.max(harvest_capacities)  # A large constant
            model.addCons(quicksum(flow_vars[(i, j)] for j in range(port_nodes) if (i, j) in arcs) >= min_harvest_levels[i] * arc_vars[i], f"min_harvest_level_{i}_big_M")

        # Carbon emissions constraint related to the flow
        for i, j in arcs:
            model.addCons(carbon_vars[(i, j)] == flow_vars[(i, j)] * carbon_emission_rates[i][j], f"carbon_emission_{i}_{j}")

        # Port demand satisfaction constraints
        for j in range(port_nodes):
            model.addCons(quicksum(flow_vars[(i, j)] for i in range(storage_nodes) if (i, j) in arcs) >= port_demand[j], f"port_demand_{j}")

        # Objective function: minimize total cost
        transportation_cost_expr = quicksum(flow_vars[(i, j)] * shipping_costs[i, j] for (i, j) in arcs)
        fixed_cost_expr = quicksum(arc_vars[i] * fixed_costs[i] for i in range(storage_nodes))
        carbon_cost_expr = quicksum(carbon_vars[(i, j)] for (i, j) in arcs)  # New carbon cost
        total_cost = transportation_cost_expr + fixed_cost_expr + carbon_cost_expr
        
        ### new constraints and variables and objective code ends here
        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'storage_nodes': 375,
        'port_nodes': 112,
    }

    agric_problem = AgriculturalDistributionNetwork(parameters, seed=seed)
    instance = agric_problem.generate_instance()
    solve_status, solve_time = agric_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")