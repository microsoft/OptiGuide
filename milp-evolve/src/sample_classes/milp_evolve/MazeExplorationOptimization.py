import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MazeExplorationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_robots > 0 and self.n_cells > 0
        assert self.min_exploration_cost >= 0 and self.max_exploration_cost >= self.min_exploration_cost
        assert self.min_detection_cost >= 0 and self.max_detection_cost >= self.min_detection_cost
        assert self.min_battery_capacity > 0 and self.max_battery_capacity >= self.min_battery_capacity
        assert self.min_critical_items >= 0 and self.max_critical_items >= self.min_critical_items

        exploration_costs = np.random.randint(self.min_exploration_cost, self.max_exploration_cost + 1, self.n_robots)
        detection_costs = np.random.randint(self.min_detection_cost, self.max_detection_cost + 1, (self.n_robots, self.n_cells))
        battery_capacities = np.random.randint(self.min_battery_capacity, self.max_battery_capacity + 1, self.n_robots)
        critical_items = np.random.randint(self.min_critical_items, self.max_critical_items + 1, self.n_cells)
        
        energy_limits = np.random.randint(200, 500, self.n_robots) # in energy units
        sensor_ranges = np.random.randint(20, 50, self.n_robots) # in cell units
        priority_cells = np.random.choice([0, 1], self.n_cells, p=[0.7, 0.3]) # 30% high-priority cells
        
        return {
            "exploration_costs": exploration_costs,
            "detection_costs": detection_costs,
            "battery_capacities": battery_capacities,
            "critical_items": critical_items,
            "energy_limits": energy_limits,
            "sensor_ranges": sensor_ranges,
            "priority_cells": priority_cells
        }

    def solve(self, instance):
        exploration_costs = instance['exploration_costs']
        detection_costs = instance['detection_costs']
        battery_capacities = instance['battery_capacities']
        critical_items = instance['critical_items']
        energy_limits = instance['energy_limits']
        sensor_ranges = instance['sensor_ranges']
        priority_cells = instance['priority_cells']

        model = Model("MazeExplorationOptimization")
        n_robots = len(exploration_costs)
        n_cells = len(critical_items)
        
        robot_vars = {r: model.addVar(vtype="B", name=f"Robot_{r}") for r in range(n_robots)}
        cell_exploration_vars = {(r, c): model.addVar(vtype="C", name=f"Cell_{r}_Cell_{c}") for r in range(n_robots) for c in range(n_cells)}
        unmet_detection_vars = {c: model.addVar(vtype="C", name=f"Unmet_Cell_{c}") for c in range(n_cells)}
        
        coordinate_vars = {r: model.addVar(vtype="B", name=f"Coordinate_{r}") for r in range(n_robots)}

        model.setObjective(
            quicksum(exploration_costs[r] * robot_vars[r] for r in range(n_robots)) +
            quicksum(detection_costs[r][c] * cell_exploration_vars[r, c] for r in range(n_robots) for c in range(n_cells)) +
            quicksum(1000 * unmet_detection_vars[c] for c in range(n_cells) if priority_cells[c] == 1),
            "minimize"
        )

        # Critical item detection satisfaction
        for c in range(n_cells):
            model.addCons(quicksum(cell_exploration_vars[r, c] for r in range(n_robots)) + unmet_detection_vars[c] == critical_items[c], f"Neighborhood_Detection_{c}")
        
        # Battery limits for each robot
        for r in range(n_robots):
            model.addCons(quicksum(cell_exploration_vars[r, c] for c in range(n_cells)) <= battery_capacities[r] * robot_vars[r], f"Maze_Battery_Limit_{r}")

        # Cell exploration only if robot is active
        for r in range(n_robots):
            for c in range(n_cells):
                model.addCons(cell_exploration_vars[r, c] <= critical_items[c] * robot_vars[r], f"Active_Robot_Constraint_{r}_{c}")

        # Energy consumption constraints
        for r in range(n_robots):
            model.addCons(quicksum(cell_exploration_vars[r, c] for c in range(n_cells)) <= energy_limits[r], f"Energy_Limit_{r}")
        
        # Sensor range for each robot
        for r in range(n_robots):
            model.addCons(quicksum(cell_exploration_vars[r, c] * 0.5 for c in range(n_cells)) <= sensor_ranges[r], f"Sensor_Range_{r}")

        # Priority cell detection
        for c in range(n_cells):
            if priority_cells[c] == 1:
                model.addCons(quicksum(cell_exploration_vars[r, c] for r in range(n_robots)) + unmet_detection_vars[c] >= critical_items[c], f"Priority_Detection_{c}")
        
        # Ensure coordinate assignment for active robots
        for r in range(n_robots):
            model.addCons(robot_vars[r] == coordinate_vars[r], f"Coordinate_Assignment_{r}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_robots': 100,
        'n_cells': 300,
        'min_exploration_cost': 3000,
        'max_exploration_cost': 3000,
        'min_detection_cost': 150,
        'max_detection_cost': 750,
        'min_battery_capacity': 750,
        'max_battery_capacity': 750,
        'min_critical_items': 3,
        'max_critical_items': 80,
    }

    optimizer = MazeExplorationOptimization(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")