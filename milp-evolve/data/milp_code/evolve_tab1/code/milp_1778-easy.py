import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnergyEfficientHCANScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

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

        G = nx.erdos_renyi_graph(n=self.city_nodes, p=self.zone_prob, directed=True, seed=self.seed)
        for node in G.nodes:
            G.nodes[node]['patients'] = np.random.randint(10, 200)
        for u, v in G.edges:
            G[u][v]['visit_time'] = np.random.randint(1, 3)
            G[u][v]['capacity'] = np.random.randint(5, 15)

        healthcare_cap = {node: np.random.randint(20, 100) for node in G.nodes}
        shift_cost = {(u, v): np.random.uniform(1.0, 5.0) for u, v in G.edges}
        daily_appointments = [(zone, np.random.uniform(80, 400)) for zone in list(nx.find_cliques(G.to_undirected()))]

        temp_control_costs = {(u, v): np.random.uniform(5.0, 15.0) for u, v in G.edges}
        max_temp = 8

        instance = {
            "task_times": task_times,
            "power_costs": power_costs,
            "task_deadlines": task_deadlines,
            "machine_capacities": machine_capacities,
            "energy_consumptions": energy_consumptions,
            'G': G,
            'healthcare_cap': healthcare_cap,
            'shift_cost': shift_cost,
            'daily_appointments': daily_appointments,
            'temp_control_costs': temp_control_costs,
            'max_temp': max_temp,
        }
        return instance

    def solve(self, instance):
        task_times = instance['task_times']
        power_costs = instance['power_costs']
        task_deadlines = instance['task_deadlines']
        machine_capacities = instance['machine_capacities']
        energy_consumptions = instance['energy_consumptions']

        G = instance['G']
        healthcare_cap = instance['healthcare_cap']
        shift_cost = instance['shift_cost']
        daily_appointments = instance['daily_appointments']
        temp_control_costs = instance['temp_control_costs']
        max_temp = instance['max_temp']

        model = Model("EnergyEfficientHCANScheduling")
        n_tasks = len(task_times)
        n_machines = len(power_costs)

        # Decision variables
        start_times = {t: model.addVar(vtype="I", name=f"StartTime_{t}") for t in range(n_tasks)}
        machine_assignments = {(m, t): model.addVar(vtype="B", name=f"Assignment_{m}_{t}") for m in range(n_machines) for t in range(n_tasks)}
        temperature_control_vars = {f"TempControl{u}_{v}": model.addVar(vtype="B", name=f"TempControl{u}_{v}") for u, v in G.edges}
        
        # Objective: Minimize total energy consumption
        model.setObjective(
            quicksum(energy_consumptions[m, t] * machine_assignments[(m, t)] * task_times[t] * power_costs[m] for m in range(n_machines) for t in range(n_tasks)) +
            quicksum(temp_control_costs[(u, v)] * temperature_control_vars[f"TempControl{u}_{v}"] for u, v in G.edges),
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

        # Temperature control constraints
        for u, v in G.edges:
            model.addCons(temperature_control_vars[f"TempControl{u}_{v}"] * max_temp >= self.min_temp, f"TemperatureControl_{u}_{v}")

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
        'city_nodes': 50,
        'zone_prob': 0.2,
        'min_temp': 4,
    }

    optimizer = EnergyEfficientHCANScheduling(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")