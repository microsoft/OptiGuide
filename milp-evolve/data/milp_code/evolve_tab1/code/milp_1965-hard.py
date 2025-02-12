import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class NeighborhoodEnergyDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_edges = int(self.n_substations * self.n_areas * self.density)
        
        # Load demands and substation capacities
        residential_loads = np.random.randint(self.min_load, self.max_load, size=self.n_areas)
        substation_capacities = np.random.randint(self.min_capacity, self.max_capacity, size=self.n_substations)
        
        # Energy efficiency and cost parameters
        energy_efficiency = np.random.uniform(self.min_efficiency, self.max_efficiency, size=(self.n_substations, self.n_areas))
        substation_costs = np.random.randint(self.cost_low, self.cost_high, size=self.n_substations)
        
        res = {
            'residential_loads': residential_loads,
            'substation_capacities': substation_capacities,
            'energy_efficiency': energy_efficiency,
            'substation_costs': substation_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        residential_loads = instance['residential_loads']
        substation_capacities = instance['substation_capacities']
        energy_efficiency = instance['energy_efficiency']
        substation_costs = instance['substation_costs']

        model = Model("NeighborhoodEnergyDistribution")
        substation_vars = {}
        distribution_vars = {}
        energy_loss_vars = {}

        # Create variables and set objectives
        for j in range(self.n_substations):
            substation_vars[j] = model.addVar(vtype="B", name=f"Substation_{j}", obj=substation_costs[j])

        for area in range(self.n_areas):
            for sub in range(self.n_substations):
                distribution_vars[(area, sub)] = model.addVar(vtype="C", name=f"Area_{area}_Substation_{sub}", obj=energy_efficiency[sub][area])

        for sub in range(self.n_substations):
            energy_loss_vars[sub] = model.addVar(vtype="C", name=f"Energy_Loss_{sub}")

        # Ensure each area's load demand is met
        for area in range(self.n_areas):
            model.addCons(quicksum(distribution_vars[(area, sub)] for sub in range(self.n_substations)) >= residential_loads[area], f"Load_{area}_Demand")

        # Substation capacity constraints
        for sub in range(self.n_substations):
            model.addCons(quicksum(distribution_vars[(area, sub)] for area in range(self.n_areas)) <= substation_capacities[sub], f"Substation_{sub}_Capacity")
            model.addCons(energy_loss_vars[sub] == quicksum((1 - energy_efficiency[sub][area]) * distribution_vars[(area, sub)] for area in range(self.n_areas)), f"Energy_Loss_{sub}")
            model.addCons(energy_loss_vars[sub] <= substation_capacities[sub] * substation_vars[sub], f"Substation_{sub}_Active_Capacity")

        # Objective: Minimize total cost including energy distribution and substation operation costs
        objective_expr = quicksum(substation_costs[j] * substation_vars[j] for j in range(self.n_substations)) + \
                         quicksum(energy_efficiency[sub][area] * distribution_vars[(area, sub)] for sub in range(self.n_substations) for area in range(self.n_areas))

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_areas': 75,
        'n_substations': 210,
        'density': 0.82,
        'min_load': 200,
        'max_load': 1200,
        'min_capacity': 1000,
        'max_capacity': 1200,
        'min_efficiency': 0.78,
        'max_efficiency': 0.83,
        'cost_low': 600,
        'cost_high': 2000,
    }

    problem = NeighborhoodEnergyDistribution(parameters, seed=seed)
    instance = problem.generate_instance()
    solve_status, solve_time = problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")