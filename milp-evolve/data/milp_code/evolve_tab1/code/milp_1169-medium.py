import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DroneDeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_drones > 0 and self.n_tasks > 0
        assert self.min_operational_cost >= 0 and self.max_operational_cost >= self.min_operational_cost
        assert self.min_delivery_cost >= 0 and self.max_delivery_cost >= self.min_delivery_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        assert self.min_task_weight >= 0 and self.max_task_weight >= self.min_task_weight

        operational_costs = np.random.randint(self.min_operational_cost, self.max_operational_cost + 1, self.n_drones)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost + 1, (self.n_drones, self.n_tasks))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_drones)
        task_weights = np.random.randint(self.min_task_weight, self.max_task_weight + 1, self.n_tasks)
        priority_tasks = np.random.choice([0, 1], self.n_tasks, p=[0.7, 0.3]) # 30% high-priority tasks
        
        return {
            "operational_costs": operational_costs,
            "delivery_costs": delivery_costs,
            "capacities": capacities,
            "task_weights": task_weights,
            "priority_tasks": priority_tasks
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        delivery_costs = instance['delivery_costs']
        capacities = instance['capacities']
        task_weights = instance['task_weights']
        priority_tasks = instance['priority_tasks']

        model = Model("DroneDeliveryOptimization")
        n_drones = len(operational_costs)
        n_tasks = len(task_weights)
        
        drone_vars = {d: model.addVar(vtype="B", name=f"Drone_{d}") for d in range(n_drones)}
        task_assignment_vars = {(d, t): model.addVar(vtype="C", name=f"Task_{d}_Task_{t}") for d in range(n_drones) for t in range(n_tasks)}
        unmet_task_vars = {t: model.addVar(vtype="C", name=f"Unmet_Task_{t}") for t in range(n_tasks)}

        model.setObjective(
            quicksum(operational_costs[d] * drone_vars[d] for d in range(n_drones)) +
            quicksum(delivery_costs[d][t] * task_assignment_vars[d, t] for d in range(n_drones) for t in range(n_tasks)) +
            quicksum(1000 * unmet_task_vars[t] for t in range(n_tasks) if priority_tasks[t] == 1),
            "minimize"
        )

        # Task weight satisfaction
        for t in range(n_tasks):
            model.addCons(quicksum(task_assignment_vars[d, t] for d in range(n_drones)) + unmet_task_vars[t] == task_weights[t], f"Task_Weight_Satisfaction_{t}")
        
        # Capacity limits for each drone
        for d in range(n_drones):
            model.addCons(quicksum(task_assignment_vars[d, t] for t in range(n_tasks)) <= capacities[d] * drone_vars[d], f"Drone_Capacity_{d}")

        # Task assignment only if drone is operational
        for d in range(n_drones):
            for t in range(n_tasks):
                model.addCons(task_assignment_vars[d, t] <= task_weights[t] * drone_vars[d], f"Operational_Constraint_{d}_{t}")
        
        # Priority task constraints
        for t in range(n_tasks):
            if priority_tasks[t] == 1:
                model.addCons(quicksum(task_assignment_vars[d, t] for d in range(n_drones)) + unmet_task_vars[t] >= task_weights[t], f"Priority_Task_{t}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_drones': 250,
        'n_tasks': 300,
        'min_operational_cost': 2000,
        'max_operational_cost': 5000,
        'min_delivery_cost': 300,
        'max_delivery_cost': 1500,
        'min_capacity': 100,
        'max_capacity': 100,
        'min_task_weight': 9,
        'max_task_weight': 100,
    }

    optimizer = DroneDeliveryOptimization(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")