import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class EdgeComputingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_nodes > 0 and self.n_tasks > 0
        assert self.min_comm_cost >= 0 and self.max_comm_cost >= self.min_comm_cost
        assert self.min_node_energy_cost >= 0 and self.max_node_energy_cost >= self.min_node_energy_cost
        assert self.min_node_capacity > 0 and self.max_node_capacity >= self.min_node_capacity
        assert self.min_latency >= 0 and self.max_latency >= self.min_latency
        
        node_energy_costs = np.random.randint(self.min_node_energy_cost, self.max_node_energy_cost + 1, self.n_nodes)
        comm_costs = np.random.randint(self.min_comm_cost, self.max_comm_cost + 1, (self.n_nodes, self.n_tasks))
        latencies = np.random.randint(self.min_latency, self.max_latency + 1, (self.n_nodes, self.n_tasks))
        capacities = np.random.randint(self.min_node_capacity, self.max_node_capacity + 1, self.n_nodes)
        task_demand = np.random.randint(1, 10, self.n_tasks)
        
        return {
            "node_energy_costs": node_energy_costs,
            "comm_costs": comm_costs,
            "latencies": latencies,
            "capacities": capacities,
            "task_demand": task_demand,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        node_energy_costs = instance['node_energy_costs']
        comm_costs = instance['comm_costs']
        latencies = instance['latencies']
        capacities = instance['capacities']
        task_demand = instance['task_demand']
        
        model = Model("EdgeComputingOptimization")
        n_nodes = len(node_energy_costs)
        n_tasks = len(task_demand)
        
        # Decision variables
        node_vars = {n: model.addVar(vtype="B", name=f"Node_{n}") for n in range(n_nodes)}
        task_vars = {(n, t): model.addVar(vtype="B", name=f"Node_{n}_Task_{t}") for n in range(n_nodes) for t in range(n_tasks)}
        total_latency = model.addVar(vtype="C", lb=0, name="TotalLatency")
        
        # Objective: minimize the total cost (node energy + comm + latency penalties)
        model.setObjective(quicksum(node_energy_costs[n] * node_vars[n] for n in range(n_nodes)) +
                           quicksum(comm_costs[n, t] * task_vars[n, t] for n in range(n_nodes) for t in range(n_tasks)) +
                           50 * total_latency, "minimize")
        
        # Constraints: Each task demand is met
        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[n, t] for n in range(n_nodes)) == 1, f"Task_{t}_Demand")
        
        # Constraints: Only active nodes can process tasks
        for n in range(n_nodes):
            for t in range(n_tasks):
                model.addCons(task_vars[n, t] <= node_vars[n], f"Node_{n}_Process_{t}")
        
        # Constraints: Nodes cannot exceed their capacity
        for n in range(n_nodes):
            model.addCons(quicksum(task_demand[t] * task_vars[n, t] for t in range(n_tasks)) <= capacities[n], f"Node_{n}_Capacity")
        
        # Constraints: Total latency should not exceed given limit (arbitrary fixed value)
        for n in range(n_nodes):
            model.addCons(quicksum(latencies[n, t] * task_vars[n, t] for t in range(n_tasks)) <= total_latency, f"Node_{n}_Latency")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 30,
        'n_tasks': 30,
        'min_comm_cost': 1,
        'max_comm_cost': 50,
        'min_node_energy_cost': 750,
        'max_node_energy_cost': 3000,
        'min_node_capacity': 10,
        'max_node_capacity': 2500,
        'min_latency': 2,
        'max_latency': 10,
    }

    edge_computing_optimizer = EdgeComputingOptimization(parameters, seed=42)
    instance = edge_computing_optimizer.generate_instance()
    solve_status, solve_time, objective_value = edge_computing_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")