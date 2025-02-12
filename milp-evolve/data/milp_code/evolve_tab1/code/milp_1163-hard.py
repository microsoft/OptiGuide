import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ManufacturingPlantOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_workstations > 0 and self.n_tasks >= self.n_workstations
        assert self.min_raw_cost >= 0 and self.max_raw_cost >= self.min_raw_cost
        assert self.min_energy_cost >= 0 and self.max_energy_cost >= self.min_energy_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity

        raw_costs = np.random.randint(self.min_raw_cost, self.max_raw_cost + 1, self.n_workstations)
        energy_costs = np.random.normal(self.mean_energy_cost, self.stddev_energy_cost, (self.n_workstations, self.n_tasks)).astype(int)
        energy_costs = np.clip(energy_costs, self.min_energy_cost, self.max_energy_cost)
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_workstations)

        market_demand = {t: np.random.uniform(self.min_demand, self.max_demand) for t in range(self.n_tasks)}

        G = nx.erdos_renyi_graph(n=self.n_tasks, p=self.route_prob, directed=True, seed=self.seed)
        for node in G.nodes:
            G.nodes[node]['processing_rate'] = np.random.uniform(0.8, 1.2)
        for u, v in G.edges:
            G[u][v]['transition_time'] = np.random.randint(5, 15)

        maintenance_schedule = np.random.uniform(self.min_maintenance_time, self.max_maintenance_time, self.n_workstations)
        defect_rate = np.random.uniform(self.min_defect_rate, self.max_defect_rate, self.n_workstations)

        energy_limits = np.random.uniform(self.min_energy_limit, self.max_energy_limit, self.n_workstations)
        waste_limits = np.random.uniform(self.min_waste_limit, self.max_waste_limit, self.n_workstations)

        return {
            "raw_costs": raw_costs,
            "energy_costs": energy_costs,
            "capacities": capacities,
            "market_demand": market_demand,
            "G": G,
            "maintenance_schedule": maintenance_schedule,
            "defect_rate": defect_rate,
            "energy_limits": energy_limits,
            "waste_limits": waste_limits,
        }

    def solve(self, instance):
        raw_costs = instance['raw_costs']
        energy_costs = instance['energy_costs']
        capacities = instance['capacities']
        market_demand = instance['market_demand']
        G = instance['G']
        maintenance_schedule = instance['maintenance_schedule']
        defect_rate = instance['defect_rate']
        energy_limits = instance['energy_limits']
        waste_limits = instance['waste_limits']

        model = Model("ManufacturingPlantOptimization")
        n_workstations = len(raw_costs)
        n_tasks = len(energy_costs[0])

        # Decision variables
        workstation_vars = {w: model.addVar(vtype="B", name=f"Workstation_{w}") for w in range(n_workstations)}
        allocation_vars = {(w, t): model.addVar(vtype="B", name=f"Workstation_{w}_Task_{t}") for w in range(n_workstations) for t in range(n_tasks)}
        dynamic_capacity_vars = {w: model.addVar(vtype="C", name=f"DynamicCapacity_{w}") for w in range(n_workstations)}
        demand_shift_vars = {t: model.addVar(vtype="C", name=f"DemandShift_{t}") for t in range(n_tasks)}

        # Objective: Maximize total production efficiency while minimizing costs and defects
        model.setObjective(
            quicksum(market_demand[t] * quicksum(allocation_vars[w, t] for w in range(n_workstations)) for t in range(n_tasks)) -
            quicksum(raw_costs[w] * workstation_vars[w] for w in range(n_workstations)) -
            quicksum(energy_costs[w][t] * allocation_vars[w, t] for w in range(n_workstations) for t in range(n_tasks)) -
            quicksum(defect_rate[w] * allocation_vars[w, t] for w in range(n_workstations) for t in range(n_tasks)),
            "maximize"
        )

        # Constraints: Each task is performed by exactly one workstation
        for t in range(n_tasks):
            model.addCons(quicksum(allocation_vars[w, t] for w in range(n_workstations)) == 1, f"Task_{t}_Assignment")

        # Constraints: Only active workstations can perform tasks
        for w in range(n_workstations):
            for t in range(n_tasks):
                model.addCons(allocation_vars[w, t] <= workstation_vars[w], f"Workstation_{w}_Task_{t}_Service")

        # Constraints: Workstations cannot exceed their dynamic capacity
        for w in range(n_workstations):
            model.addCons(quicksum(allocation_vars[w, t] for t in range(n_tasks)) <= dynamic_capacity_vars[w], f"Workstation_{w}_DynamicCapacity")

        # Dynamic Capacity Constraints based on maintenance and market demand
        for w in range(n_workstations):
            model.addCons(dynamic_capacity_vars[w] == (capacities[w] - maintenance_schedule[w]), f"DynamicCapacity_{w}")

        # Maintenance Downtime Constraints
        for w in range(n_workstations):
            model.addCons(workstation_vars[w] * maintenance_schedule[w] <= self.max_maintenance_time, f"MaintenanceDowntime_{w}")

        # Quality Control Constraints
        for w in range(n_workstations):
            model.addCons(defect_rate[w] * quicksum(allocation_vars[w, t] for t in range(n_tasks)) <= self.max_defects, f"QualityControl_{w}")

        # Environmental Constraints
        for w in range(n_workstations):
            model.addCons(
                quicksum(energy_costs[w][t] * allocation_vars[w, t] for t in range(n_tasks)) <= energy_limits[w], f"EnergyLimit_{w}"
            )
            model.addCons(
                quicksum(energy_costs[w][t] / 1000 * allocation_vars[w, t] for t in range(n_tasks)) <= waste_limits[w], f"WasteLimit_{w}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_workstations': 24,
        'n_tasks': 500,
        'min_raw_cost': 300,
        'max_raw_cost': 1000,
        'mean_energy_cost': 50,
        'stddev_energy_cost': 160,
        'min_energy_cost': 20,
        'max_energy_cost': 900,
        'min_capacity': 150,
        'max_capacity': 1400,
        'route_prob': 0.6,
        'min_demand': 90,
        'max_demand': 2000,
        'min_maintenance_time': 5,
        'max_maintenance_time': 135,
        'min_defect_rate': 0.38,
        'max_defect_rate': 0.38,
        'min_energy_limit': 3000,
        'max_energy_limit': 5000,
        'min_waste_limit': 350,
        'max_waste_limit': 1200,
        'max_defects': 80,
    }

    optimizer = ManufacturingPlantOptimization(parameters, seed=42)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")