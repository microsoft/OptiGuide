import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CloudNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def get_instance(self):
        assert self.n_devices > 0 and self.n_clients > 0
        assert self.min_delay >= 0 and self.max_delay >= self.min_delay
        assert self.min_device_cost >= 0 and self.max_device_cost >= self.min_device_cost
        assert self.min_device_capacity > 0 and self.max_device_capacity >= self.min_device_capacity
        
        device_costs = np.random.randint(self.min_device_cost, self.max_device_cost + 1, self.n_devices)
        network_delays = np.random.randint(self.min_delay, self.max_delay + 1, (self.n_devices, self.n_clients))
        capacities = np.random.randint(self.min_device_capacity, self.max_device_capacity + 1, self.n_devices)
        demands = np.random.randint(1, 10, self.n_clients)
        run_times = np.random.randint(self.min_run_time, self.max_run_time + 1, self.n_devices)
        security_levels = np.random.randint(self.min_security_level, self.max_security_level + 1, self.n_devices)
        
        return {
            "device_costs": device_costs,
            "network_delays": network_delays,
            "capacities": capacities,
            "demands": demands,
            "run_times": run_times,
            "security_levels": security_levels,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        device_costs = instance['device_costs']
        network_delays = instance['network_delays']
        capacities = instance['capacities']
        demands = instance['demands']
        run_times = instance['run_times']
        security_levels = instance['security_levels']
        
        model = Model("CloudNetworkOptimization")
        n_devices = len(device_costs)
        n_clients = len(network_delays[0])
        
        # Decision variables
        device_vars = {d: model.addVar(vtype="B", name=f"Device_{d}") for d in range(n_devices)}
        connectivity_vars = {(d, c): model.addVar(vtype="B", name=f"Device_{d}_Client_{c}") for d in range(n_devices) for c in range(n_clients)}
        run_time_vars = {d: model.addVar(vtype="I", lb=0, name=f"RunTime_{d}") for d in range(n_devices)}
        security_level_vars = {d: model.addVar(vtype="I", lb=0, name=f"SecurityLevel_{d}") for d in range(n_devices)}
        
        # Objective: minimize the total network delay while maintaining cost efficiency and security level
        model.setObjective(
            quicksum(device_costs[d] * device_vars[d] for d in range(n_devices)) +
            quicksum(network_delays[d, c] * connectivity_vars[d, c] for d in range(n_devices) for c in range(n_clients)) +
            quicksum(run_times[d] * run_time_vars[d] for d in range(n_devices)) +
            quicksum(security_levels[d] * security_level_vars[d] for d in range(n_devices)) -
            50 * quicksum(connectivity_vars[d, c] for d in range(n_devices) for c in range(n_clients)) 
            , "minimize"
        )
        
        # Constraints: Each client demand is met by exactly one device
        for c in range(n_clients):
            model.addCons(quicksum(connectivity_vars[d, c] for d in range(n_devices)) == 1, f"Client_{c}_Demand")
        
        # Constraints: Only connected devices can serve clients
        for d in range(n_devices):
            for c in range(n_clients):
                model.addCons(connectivity_vars[d, c] <= device_vars[d], f"Device_{d}_Serve_{c}")
        
        # Constraints: Devices cannot exceed their capacity
        for d in range(n_devices):
            model.addCons(quicksum(demands[c] * connectivity_vars[d, c] for c in range(n_clients)) <= capacities[d], f"Device_{d}_Capacity")
        
        # Constraints: Devices must respect their running time and security levels
        for d in range(n_devices):
            model.addCons(run_time_vars[d] == run_times[d], f"Device_{d}_RunTime")
            model.addCons(security_level_vars[d] == security_levels[d], f"Device_{d}_SecurityLevel")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_devices': 50,
        'n_clients': 180,
        'min_delay': 250,
        'max_delay': 400,
        'min_device_cost': 3000,
        'max_device_cost': 8000,
        'min_device_capacity': 1000,
        'max_device_capacity': 2000,
        'min_run_time': 6,
        'max_run_time': 32,
        'min_security_level': 420,
        'max_security_level': 1200,
    }
    ### given parameter code ends here
    ### new parameter code ends here
    
    cloud_network_optimizer = CloudNetworkOptimization(parameters, seed=42)
    instance = cloud_network_optimizer.get_instance()
    solve_status, solve_time, objective_value = cloud_network_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")