import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class VMPlacement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        # Generate random resource demands for VMs and capacities for PMs
        vm_demands = np.random.randint(1, self.max_demand, size=(self.n_vms, self.n_resources))
        pm_capacities = np.random.randint(self.max_capacity // 2, self.max_capacity, size=(self.n_pms, self.n_resources))

        # Randomly generate the bandwidth requirements and energy costs
        bandwidth = np.random.randint(1, self.max_bandwidth, size=(self.n_vms, self.n_vms))
        energy_costs = np.random.rand(self.n_pms) * self.electricity_rate

        res = {
            'vm_demands': vm_demands,
            'pm_capacities': pm_capacities,
            'bandwidth': bandwidth,
            'energy_costs': energy_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        vm_demands = instance['vm_demands']
        pm_capacities = instance['pm_capacities']
        bandwidth = instance['bandwidth']
        energy_costs = instance['energy_costs']

        model = Model("VMPlacement")
        var_vm_pm = {}
        pm_load = {}

        # Create variables
        for p in range(self.n_pms):
            pm_load[p] = model.addVar(vtype="C", name=f"PM_load_{p}")
            for i in range(self.n_vms):
                var_vm_pm[i, p] = model.addVar(vtype="B", name=f"vm_p_{i}_{p}")

        # Add constraints to ensure each VM is placed exactly once
        for i in range(self.n_vms):
            model.addCons(quicksum(var_vm_pm[i, p] for p in range(self.n_pms)) == 1, f"vm_placed_{i}")

        # Capacity constraints
        for p in range(self.n_pms):
            for r in range(self.n_resources):
                model.addCons(
                    quicksum(vm_demands[i, r] * var_vm_pm[i, p] for i in range(self.n_vms)) <= pm_capacities[p, r], 
                    f"capacity_{p}_{r}"
                )
        
        # Compute PM load based on VMs placed
        for p in range(self.n_pms):
            model.addCons(
                pm_load[p] == quicksum(vm_demands[i, 0] * var_vm_pm[i, p] for i in range(self.n_vms)), 
                f"compute_load_{p}"
            )

        # Objective: Minimize total energy consumption
        objective_expr = quicksum(pm_load[p] * energy_costs[p] for p in range(self.n_pms))
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_vms': 100,
        'n_pms': 50,
        'n_resources': 3,
        'max_demand': 10,
        'max_capacity': 100,
        'max_bandwidth': 10,
        'electricity_rate': 0.1,
    }

    vm_placement_problem = VMPlacement(parameters, seed=seed)
    instance = vm_placement_problem.generate_instance()
    solve_status, solve_time = vm_placement_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")