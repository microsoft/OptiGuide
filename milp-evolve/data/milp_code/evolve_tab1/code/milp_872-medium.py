import random
import time
import numpy as np
from pyscipopt import Model, quicksum, multidict

class DistributedServerFarmOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_machines = random.randint(self.min_machines, self.max_machines)
        num_clients = random.randint(self.min_clients, self.max_clients)
        
        # Cost matrices
        construction_cost = np.random.randint(10000, 30000, size=num_machines)
        operating_cost = np.random.randint(500, 2000, size=num_machines)
        core_cost = np.random.randint(50, 150, size=(num_clients, num_machines))
        
        # Client requirements
        client_core_demand = np.random.randint(10, 50, size=num_clients)
        
        # Demand uncertainty
        client_core_demand_uncertainty = np.random.randint(2, 5, size=num_clients)
        
        # Machines' capacities
        machine_core_capacity = np.random.randint(200, 1000, size=num_machines)
        
        # Bandwidth limits
        core_bandwidth_limit = np.random.randint(100, 500)
        
        # Budget constraints
        total_budget = np.random.randint(100000, 200000)
        
        # Hardware costs
        num_hardwares = 3
        hardware_cost = np.random.randint(500, 3000, size=num_hardwares)
        hardware_capacity = np.random.randint(50, 200, size=num_hardwares)
        max_hardwares_per_machine = np.random.randint(1, num_hardwares + 1, size=num_machines)

        res = {
            'num_machines': num_machines,
            'num_clients': num_clients,
            'construction_cost': construction_cost,
            'operating_cost': operating_cost,
            'core_cost': core_cost,
            'client_core_demand': client_core_demand,
            'client_core_demand_uncertainty': client_core_demand_uncertainty,
            'machine_core_capacity': machine_core_capacity,
            'core_bandwidth_limit': core_bandwidth_limit,
            'total_budget': total_budget,
            'num_hardwares': num_hardwares,
            'hardware_cost': hardware_cost,
            'hardware_capacity': hardware_capacity,
            'max_hardwares_per_machine': max_hardwares_per_machine
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_machines = instance['num_machines']
        num_clients = instance['num_clients']
        construction_cost = instance['construction_cost']
        operating_cost = instance['operating_cost']
        core_cost = instance['core_cost']
        client_core_demand = instance['client_core_demand']
        client_core_demand_uncertainty = instance['client_core_demand_uncertainty']
        machine_core_capacity = instance['machine_core_capacity']
        core_bandwidth_limit = instance['core_bandwidth_limit']
        total_budget = instance['total_budget']
        num_hardwares = instance['num_hardwares']
        hardware_cost = instance['hardware_cost']
        hardware_capacity = instance['hardware_capacity']
        max_hardwares_per_machine = instance['max_hardwares_per_machine']

        model = Model("DistributedServerFarmOptimization")

        # Variables
        MachineConstruction = {j: model.addVar(vtype="B", name=f"MachineConstruction_{j}") for j in range(num_machines)}
        CoreAllocation = {(i, j): model.addVar(vtype="I", name=f"CoreAllocation_{i}_{j}") for i in range(num_clients) for j in range(num_machines)}

        # Piecewise linear operating cost parameter k for each machine
        pwl_segments = {(j, k): (random.uniform(0.8, 1.2), round((operating_cost[j] / num_machines) * (k+1))) for j in range(num_machines) for k in range(3)}
        UtilizationSegment = {(j, k): model.addVar(vtype="B", name=f"UtilizationSegment_{j}_{k}") for j in range(num_machines) for k in range(3)}

        # New variables for hardware
        HardwareInstallation = {(j, k): model.addVar(vtype="B", name=f"HardwareInstallation_{j}_{k}") for j in range(num_machines) for k in range(num_hardwares)}
        NetworkNodeAvailable = {(i, j, k): model.addVar(vtype="I", name=f"NetworkNodeAvailable_{i}_{j}_{k}") for i in range(num_clients) for j in range(num_machines) for k in range(num_hardwares)}

        # Objective function: Minimize total costs including construction, core allocation, piecewise operating costs, and hardware costs
        TotalCost = quicksum(MachineConstruction[j] * construction_cost[j] for j in range(num_machines)) + \
                    quicksum(CoreAllocation[i, j] * core_cost[i, j] for i in range(num_clients) for j in range(num_machines)) + \
                    quicksum(UtilizationSegment[j, k] * pwl_segments[j, k][1] for j in range(num_machines) for k in range(3)) + \
                    quicksum(HardwareInstallation[j, k] * hardware_cost[k] for j in range(num_machines) for k in range(num_hardwares))

        model.setObjective(TotalCost, "minimize")

        ### New constraints ###

        # Robust client core demand constraints
        for i in range(num_clients):
            core_demand_minimum = client_core_demand[i] - client_core_demand_uncertainty[i]
            core_demand_maximum = client_core_demand[i] + client_core_demand_uncertainty[i]
            model.addCons(quicksum(CoreAllocation[i, j] for j in range(num_machines)) >= core_demand_minimum, name=f"core_demand_minimum_{i}")
            model.addCons(quicksum(CoreAllocation[i, j] for j in range(num_machines)) <= core_demand_maximum, name=f"core_demand_maximum_{i}")

        # New machine core capacity constraints
        for j in range(num_machines):
            model.addCons(quicksum(UtilizationSegment[j, k] for k in range(3)) == 1, name=f"utilization_segment_constraint_{j}")
            model.addCons(quicksum(CoreAllocation[i, j] for i in range(num_clients)) <= machine_core_capacity[j], name=f"machine_core_capacity_{j}")

        # Machine activity constraint
        for j in range(num_machines):
            model.addCons(MachineConstruction[j] * sum(client_core_demand) >= quicksum(CoreAllocation[i, j] for i in range(num_clients)), name=f"machine_activity_{j}")

        # New budget constraint using piecewise linear costs
        total_budget_piecewise = quicksum(UtilizationSegment[j, 0] * pwl_segments[j, 0][1] for j in range(num_machines)) + \
                                 quicksum(UtilizationSegment[j, 1] * pwl_segments[j, 1][1] for j in range(num_machines)) + \
                                 quicksum(UtilizationSegment[j, 2] * pwl_segments[j, 2][1] for j in range(num_machines))
        model.addCons(total_budget_piecewise <= total_budget, name="budget_constraint")

        # New constraints for hardware installation and usage
        for j in range(num_machines):
            model.addCons(quicksum(HardwareInstallation[j, k] for k in range(num_hardwares)) <= max_hardwares_per_machine[j], name=f"hardware_limit_{j}")
        
        for i in range(num_clients):
            for j in range(num_machines):
                model.addCons(quicksum(NetworkNodeAvailable[i, j, k] * hardware_capacity[k] for k in range(num_hardwares)) >= client_core_demand[i], name=f"hardware_meet_demand_{i}_{j}")
                model.addCons(quicksum(NetworkNodeAvailable[i, j, k] for k in range(num_hardwares)) <= quicksum(HardwareInstallation[j, k] for k in range(num_hardwares)), name=f"node_hardware_constraint_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_machines': 5,
        'max_machines': 80,
        'min_clients': 250,
        'max_clients': 3000,
    }

    optimization = DistributedServerFarmOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")