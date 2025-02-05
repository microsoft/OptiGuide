import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnergyEfficientScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_task_network(self):
        # Generate a random network of tasks
        G = nx.erdos_renyi_graph(n=self.n_tasks, p=self.er_prob, seed=self.seed)
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E_invalid.add(edge)
        return G, E_invalid

    def generate_instance(self):
        assert self.n_tasks > 0 and self.n_machines > 0
        assert self.min_task_time > 0 and self.max_task_time >= self.min_task_time
        assert self.min_power_cost > 0 and self.max_power_cost >= self.min_power_cost
        assert self.min_task_deadline > 0 and self.max_task_deadline >= self.min_task_deadline

        task_times = np.random.randint(self.min_task_time, self.max_task_time + 1, self.n_tasks)
        power_costs = np.random.randint(self.min_power_cost, self.max_power_cost + 1, self.n_machines)
        task_deadlines = np.random.randint(self.min_task_deadline, self.max_task_deadline + 1, self.n_tasks)
        machine_capacities = np.random.randint(1, self.max_machine_capacity + 1, self.n_machines)
        energy_consumptions = np.random.uniform(self.min_energy_consumption, self.max_energy_consumption, (self.n_machines, self.n_tasks))

        task_network, E_invalid = self.generate_task_network()

        instance = {
            "task_times": task_times,
            "power_costs": power_costs,
            "task_deadlines": task_deadlines,
            "machine_capacities": machine_capacities,
            "energy_consumptions": energy_consumptions,
            'task_network': task_network,
            'E_invalid': E_invalid,
        }
        return instance

    def solve(self, instance):
        task_times = instance['task_times']
        power_costs = instance['power_costs']
        task_deadlines = instance['task_deadlines']
        machine_capacities = instance['machine_capacities']
        energy_consumptions = instance['energy_consumptions']
        task_network = instance['task_network']
        E_invalid = instance['E_invalid']

        model = Model("EnergyEfficientScheduling")
        n_tasks = len(task_times)
        n_machines = len(power_costs)

        # Decision variables
        start_times = {t: model.addVar(vtype="I", name=f"StartTime_{t}") for t in range(n_tasks)}
        machine_assignments = {(m, t): model.addVar(vtype="B", name=f"Assignment_{m}_{t}") for m in range(n_machines) for t in range(n_tasks)}
        
        # Congested pairs variables
        congestion_vars = {(t1, t2): model.addVar(vtype="B", name=f"Congestion_{t1}_{t2}") for t1, t2 in E_invalid}

        # Objective: Minimize total energy consumption and congestion penalty
        model.setObjective(
            quicksum(energy_consumptions[m, t] * machine_assignments[(m, t)] * task_times[t] * power_costs[m] for m in range(n_machines) for t in range(n_tasks)) +
            quicksum(10 * congestion_vars[(t1, t2)] for t1, t2 in E_invalid),  # Penalty for congestion
            "minimize"
        )

        # Constraints
        # Each task must be assigned to exactly one machine
        for t in range(n_tasks):
            model.addCons(quicksum(machine_assignments[(m, t)] for m in range(n_machines)) == 1, f"TaskAssignment_{t}")

        # Tasks must finish before their deadlines
        for t in range(n_tasks):
            model.addCons(start_times[t] + task_times[t] <= task_deadlines[t], f"Deadline_{t}")

        # Tasks must start no earlier than time 0
        for t in range(n_tasks):
            model.addCons(start_times[t] >= 0, f"NonNegativeStart_{t}")

        # Machine capacity constraints
        for m in range(n_machines):
            model.addCons(quicksum(machine_assignments[(m, t)] * task_times[t] for t in range(n_tasks)) <= machine_capacities[m], f"MachineCapacity_{m}")

        # Congestion constraints: prevent certain pairs of tasks from being on the same machine
        for (t1, t2) in E_invalid:
            for m in range(n_machines):
                model.addCons(machine_assignments[(m, t1)] + machine_assignments[(m, t2)] <= 1 + congestion_vars[(t1, t2)], f"Congestion_{m}_{t1}_{t2}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_tasks': 300,
        'n_machines': 60,
        'min_task_time': 35,
        'max_task_time': 80,
        'min_power_cost': 15,
        'max_power_cost': 525,
        'min_task_deadline': 350,
        'max_task_deadline': 400,
        'max_machine_capacity': 700,
        'min_energy_consumption': 0.1,
        'max_energy_consumption': 8.0,
        'er_prob': 0.1,
        'alpha': 0.2,
    }

    optimizer = EnergyEfficientScheduling(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")