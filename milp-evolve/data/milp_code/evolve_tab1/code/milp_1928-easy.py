import random
import time
import numpy as np
import scipy.sparse
from pyscipopt import Model, quicksum

class EmergencyDroneLogistics:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.total_zones * self.total_drones * self.density)

        # Compute drones per zone
        indices = np.random.choice(self.total_drones, size=nnzrs)
        indices[:2 * self.total_drones] = np.repeat(np.arange(self.total_drones), 2)
        _, drones_per_zone = np.unique(indices, return_counts=True)

        # For each zone, sample random drones
        indices[:self.total_zones] = np.random.permutation(self.total_zones)
        i = 0
        indptr = [0]
        for n in drones_per_zone:
            if i >= self.total_zones:
                indices[i:i + n] = np.random.choice(self.total_zones, size=n, replace=False)
            elif i + n > self.total_zones:
                remaining_zones = np.setdiff1d(np.arange(self.total_zones), indices[i:self.total_zones], assume_unique=True)
                indices[self.total_zones:i + n] = np.random.choice(remaining_zones, size=i + n - self.total_zones, replace=False)
            i += n
            indptr.append(i)

        # Costs and data
        coverage_costs = np.random.randint(self.max_cost, size=self.total_drones) + 1
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.total_zones, self.total_drones)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        event_costs = np.random.randint(self.event_cost_low, self.event_cost_high, size=self.important_events)
        emergency_service = np.random.choice(self.total_drones, self.important_events, replace=False)
        
        energy_usage = np.random.uniform(self.energy_low, self.energy_high, self.total_drones)
        
        vehicle_costs = np.random.randint(self.min_vehicle_cost, self.max_vehicle_cost, self.total_vehicles)
        travel_times = np.random.randint(self.min_travel_time, self.max_travel_time, (self.total_vehicles, self.total_drones))
        vehicle_capacities = np.random.randint(self.min_vehicle_capacity, self.max_vehicle_capacity, self.total_vehicles)

        res = {
            'coverage_costs': coverage_costs,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'emergency_service': emergency_service,
            'event_costs': event_costs,
            'energy_usage': energy_usage,
            'vehicle_costs': vehicle_costs,
            'travel_times': travel_times,
            'vehicle_capacities': vehicle_capacities
        }

        # New instance data
        time_windows = np.random.randint(self.min_time_window, self.max_time_window, self.total_drones)
        drone_tasks = np.random.randint(1, self.max_tasks_per_drone + 1, self.total_drones)
        res.update({
            'time_windows': time_windows,
            'drone_tasks': drone_tasks
        })
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        coverage_costs = instance['coverage_costs']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        emergency_service = instance['emergency_service']
        event_costs = instance['event_costs']
        energy_usage = instance['energy_usage']
        vehicle_costs = instance['vehicle_costs']
        travel_times = instance['travel_times']
        vehicle_capacities = instance['vehicle_capacities']
        time_windows = instance['time_windows']
        drone_tasks = instance['drone_tasks']

        model = Model("EmergencyDroneLogistics")
        coverage_vars = {}
        energy_vars = {}
        vehicle_vars = {}
        vehicle_time_vars = {}
        task_vars = {}

        # Create variables and set objective
        for j in range(self.total_drones):
            coverage_vars[j] = model.addVar(vtype="B", name=f"coverage_{j}", obj=coverage_costs[j])
            energy_vars[j] = model.addVar(vtype="C", name=f"energy_{j}", obj=energy_usage[j])
            task_vars[j] = model.addVar(vtype="I", name=f"tasks_{j}")

        # Additional variables for important events
        for idx, j in enumerate(emergency_service):
            coverage_vars[j] = model.addVar(vtype="B", name=f"important_{j}", obj=event_costs[idx])

        # Vehicle routing
        for v in range(self.total_vehicles):
            vehicle_vars[v] = model.addVar(vtype="B", name=f"vehicle_{v}", obj=vehicle_costs[v])
            # Limit travel time evaluations
            for j in range(self.total_drones // 2):  # Only a subset
                vehicle_time_vars[(v, j)] = model.addVar(vtype="B", name=f"vehicle_{v}_drone_{j}")

        # Add constraints to ensure each zone is covered
        for zone in range(self.total_zones):
            drones = indices_csr[indptr_csr[zone]:indptr_csr[zone + 1]]
            model.addCons(quicksum(coverage_vars[j] for j in drones) >= 1, f"zone_coverage_{zone}")

        # Energy constraints for drones
        for j in range(self.total_drones):
            model.addCons(energy_vars[j] <= self.energy_capacity, f"energy_limit_{j}")

        # Vehicle capacity constraints
        for v in range(self.total_vehicles):
            model.addCons(quicksum(vehicle_time_vars.get((v, j), 0) for j in range(self.total_drones // 2)) <= vehicle_capacities[v], f"vehicle_capacity_{v}")

        # Logical condition constraints (new)
        for drone in range(self.total_drones):
            model.addCons(task_vars[drone] <= drone_tasks[drone], f"task_limit_{drone}")
            model.addCons(task_vars[drone] * time_windows[drone] <= self.max_time_per_drone, f"time_window_limit_{drone}")

        # Objective: Minimize cost
        obj_expr = quicksum(coverage_vars[j] * coverage_costs[j] for j in range(self.total_drones)) + \
                   quicksum(energy_vars[j] * energy_usage[j] for j in range(self.total_drones)) + \
                   quicksum(vehicle_costs[v] * vehicle_vars[v] for v in range(self.total_vehicles)) + \
                   quicksum(travel_times[v][j] * vehicle_time_vars.get((v, j), 0) for v in range(self.total_vehicles) for j in range(self.total_drones // 2))
                   
        model.setObjective(obj_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'total_zones': 187,
        'total_drones': 1800,
        'density': 0.66,
        'max_cost': 7,
        'important_events': 750,
        'event_cost_low': 93,
        'event_cost_high': 187,
        'energy_low': 0.39,
        'energy_high': 24.99,
        'total_vehicles': 7,
        'min_vehicle_cost': 1000,
        'max_vehicle_cost': 2400,
        'min_travel_time': 900,
        'max_travel_time': 1125,
        'min_vehicle_capacity': 75,
        'max_vehicle_capacity': 375,
        'energy_capacity': 25,
        'max_time_window': 120,
        'min_time_window': 30,
        'max_tasks_per_drone': 10,
        'max_time_per_drone': 720
    }

    emergency_drone_logistics = EmergencyDroneLogistics(parameters, seed=seed)
    instance = emergency_drone_logistics.generate_instance()
    solve_status, solve_time = emergency_drone_logistics.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")