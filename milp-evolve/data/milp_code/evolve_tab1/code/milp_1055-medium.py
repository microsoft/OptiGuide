import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class AutonomousDeliveryFleetOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data generation
    def generate_instance(self):
        assert self.n_vehicles > 0 and self.n_tasks > 0
        assert self.min_task_duration > 0 and self.max_task_duration >= self.min_task_duration
        assert self.min_vehicle_capacity > 0 and self.max_vehicle_capacity >= self.min_vehicle_capacity
        assert self.min_operational_cost >= 0 and self.max_operational_cost >= self.min_operational_cost
        
        task_durations = np.random.randint(self.min_task_duration, self.max_task_duration + 1, self.n_tasks)
        vehicle_capacities = np.random.randint(self.min_vehicle_capacity, self.max_vehicle_capacity + 1, self.n_vehicles)
        operational_costs = np.random.uniform(self.min_operational_cost, self.max_operational_cost, self.n_vehicles)
        task_graph = nx.erdos_renyi_graph(self.n_tasks, self.graph_density, seed=self.seed)
        
        return {
            "task_durations": task_durations,
            "vehicle_capacities": vehicle_capacities,
            "operational_costs": operational_costs,
            "task_graph": task_graph,
        }
    
    # MILP modeling
    def solve(self, instance):
        task_durations = instance['task_durations']
        vehicle_capacities = instance['vehicle_capacities']
        operational_costs = instance['operational_costs']
        task_graph = instance['task_graph']
        
        model = Model("AutonomousDeliveryFleetOptimization")
        n_vehicles = len(vehicle_capacities)
        n_tasks = len(task_durations)
        
        # Decision variables
        task_vars = { (v, t): model.addVar(vtype="B", name=f"Vehicle_{v}_Task_{t}") for v in range(n_vehicles) for t in range(n_tasks)}
        completion_vars = { t: model.addVar(vtype="C", name=f"Task_{t}_Completion") for t in range(n_tasks) }
        
        # Objective: minimize the total operational cost (fuel + maintenance)
        model.setObjective(
            quicksum(operational_costs[v] * quicksum(task_vars[(v, t)] * task_durations[t] for t in range(n_tasks)) for v in range(n_vehicles)),
            "minimize"
        )
        
        # Constraints: Each task must be assigned to exactly one vehicle
        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[(v, t)] for v in range(n_vehicles)) == 1, f"Task_{t}_Assignment")
        
        # Constraints: Vehicle capacities should not be exceeded
        for v in range(n_vehicles):
            model.addCons(quicksum(task_vars[(v, t)] * task_durations[t] for t in range(n_tasks)) <= vehicle_capacities[v], f"Vehicle_{v}_Capacity")
            
        # Constraints: Ensure task dependencies are met in the generated graph
        for u, v in task_graph.edges:
            model.addCons(completion_vars[v] >= completion_vars[u] + task_durations[u], f"Dependency_Task_{u}_{v}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        if model.getStatus() == 'optimal':
            objective_value = model.getObjVal()
        else:
            objective_value = None
        
        return model.getStatus(), end_time - start_time, objective_value

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_vehicles': 45,
        'n_tasks': 50,
        'min_task_duration': 210,
        'max_task_duration': 600,
        'min_vehicle_capacity': 300,
        'max_vehicle_capacity': 750,
        'min_operational_cost': 3.75,
        'max_operational_cost': 20.0,
        'graph_density': 0.31,
    }

    autonomous_delivery_optimizer = AutonomousDeliveryFleetOptimization(parameters, seed=42)
    instance = autonomous_delivery_optimizer.generate_instance()
    solve_status, solve_time, objective_value = autonomous_delivery_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")