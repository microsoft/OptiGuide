import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ManufacturingJobScheduling:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_machines > 0 and self.n_job_types >= self.n_machines
        assert self.min_processing_cost >= 0 and self.max_processing_cost >= self.min_processing_cost
        assert self.min_setup_cost >= 0 and self.max_setup_cost >= self.min_setup_cost
        assert self.min_maintenance_cost > 0 and self.max_maintenance_cost >= self.min_maintenance_cost

        processing_costs = np.random.randint(self.min_processing_cost, self.max_processing_cost + 1, self.n_machines)
        setup_costs = np.random.randint(self.min_setup_cost, self.max_setup_cost + 1, (self.n_machines, self.n_job_types))
        maintenance_costs = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost + 1, self.n_machines)
        profits = np.random.uniform(10, 100, self.n_job_types)
        maintenance_cost_factors = np.random.uniform(0.5, 2.0, self.n_machines).tolist()
        machine_availability = np.random.uniform(50, 200, self.n_job_types).tolist()
        processing_fluctuation = np.random.normal(1, 0.2, self.n_job_types).tolist()
        ordered_job_types = list(np.random.permutation(self.n_job_types))

        G = self.generate_random_graph()
        self.generate_revenues(G)
        cliques = self.find_maximal_cliques(G)

        job_durations = np.random.randint(1, 10, self.n_job_types)
        energy_consumption_rates = np.random.uniform(10, 30, (self.n_machines, self.n_job_types))
        
        return {
            "processing_costs": processing_costs,
            "setup_costs": setup_costs,
            "maintenance_costs": maintenance_costs,
            "profits": profits,
            "maintenance_cost_factors": maintenance_cost_factors,
            "machine_availability": machine_availability,
            "processing_fluctuation": processing_fluctuation,
            "ordered_job_types": ordered_job_types,
            "graph": G,
            "cliques": cliques,
            "job_durations": job_durations,
            "energy_consumption_rates": energy_consumption_rates,
        }
    
    def generate_random_graph(self):
        n_nodes = self.n_machines
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_revenues(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)

    def find_maximal_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def solve(self, instance):
        processing_costs = instance['processing_costs']
        setup_costs = instance['setup_costs']
        maintenance_costs = instance['maintenance_costs']
        profits = instance['profits']
        maintenance_cost_factors = instance['maintenance_cost_factors']
        machine_availability = instance['machine_availability']
        processing_fluctuation = instance['processing_fluctuation']
        ordered_job_types = instance['ordered_job_types']
        G = instance['graph']
        cliques = instance['cliques']
        job_durations = instance['job_durations']
        energy_consumption_rates = instance['energy_consumption_rates']

        model = Model("ManufacturingJobScheduling")
        n_machines = len(processing_costs)
        n_job_types = len(setup_costs[0])

        machine_usage_vars = {m: model.addVar(vtype="B", name=f"MachineUsage_{m}") for m in range(n_machines)}
        job_allocation_vars = {(m, j): model.addVar(vtype="B", name=f"Machine_{m}_JobType_{j}") for m in range(n_machines) for j in range(n_job_types)}
        changeover_vars = {(m, j1, j2): model.addVar(vtype="B", name=f"Machine_{m}_Changeover_{j1}_to_{j2}") for m in range(n_machines) for j1 in range(n_job_types) for j2 in range(n_job_types) if j1 != j2}

        maintenance_cost_vars = {m: model.addVar(vtype="C", name=f"MaintenanceCost_{m}", lb=0) for m in range(n_machines)}
        job_processing_vars = {j: model.addVar(vtype="C", name=f"JobProcessing_{j}", lb=0) for j in range(n_job_types)}
        machine_profit_vars = {m: model.addVar(vtype="B", name=f"MachineProfit_{m}") for m in range(n_machines)}

        energy_consumption_vars = {(m, j): model.addVar(vtype="C", name=f"EnergyConsumption_{m}_Job_{j}", lb=0) for m in range(n_machines) for j in range(n_job_types)}

        for u, v in G.edges:
            model.addCons(
                machine_profit_vars[u] + machine_profit_vars[v] <= 1,
                name=f"MachineProfitEdge_{u}_{v}"
            )

        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(machine_profit_vars[node] for node in clique) <= 1,
                name=f"Clique_{i}"
            )
        
        for j in range(n_job_types):
            model.addCons(quicksum(job_allocation_vars[m, j] for m in range(n_machines)) == 1, f"JobType_{j}_Assignment")
        
        for m in range(n_machines):
            for j in range(n_job_types):
                model.addCons(job_allocation_vars[m, j] <= machine_usage_vars[m], f"Machine_{m}_Service_{j}")
        
        for m in range(n_machines):
            model.addCons(quicksum(job_allocation_vars[m, j] for j in range(n_job_types)) <= maintenance_costs[m], f"Machine_{m}_Capacity")

        for m in range(n_machines):
            model.addCons(maintenance_cost_vars[m] == quicksum(job_allocation_vars[m, j] * maintenance_cost_factors[m] for j in range(n_job_types)), f"MaintenanceCost_{m}")

        for j in range(n_job_types):
            model.addCons(job_processing_vars[j] <= machine_availability[j], f"JobProcessing_{j}")

        for m in range(n_machines):
            for j1 in range(n_job_types):
                for j2 in range(n_job_types):
                    if j1 != j2:
                        model.addCons(changeover_vars[m, j1, j2] >= job_allocation_vars[m, j1] + job_allocation_vars[m, j2] - 1, f"Changeover_{m}_{j1}_{j2}")

        for j in range(n_job_types):
            model.addCons(job_processing_vars[j] <= machine_availability[j], f"JobProcessing_{j}")

        for i in range(n_job_types - 1):
            j1 = ordered_job_types[i]
            j2 = ordered_job_types[i + 1]
            for m in range(n_machines):
                model.addCons(job_allocation_vars[m, j1] + job_allocation_vars[m, j2] <= 1, f"SOS_Constraint_Machine_{m}_JobTypes_{j1}_{j2}")

        for m in range(n_machines):
            for j in range(n_job_types):
                model.addCons(
                    energy_consumption_vars[m, j] == job_allocation_vars[m, j] * energy_consumption_rates[m][j],
                    name=f"EnergyConsumption_{m}_{j}"
                )

        model.setObjective(
            quicksum(profits[j] * job_allocation_vars[m, j] * processing_fluctuation[j] for m in range(n_machines) for j in range(n_job_types)) +
            quicksum(G.nodes[m]['revenue'] * machine_profit_vars[m] for m in range(n_machines)) -
            quicksum(processing_costs[m] * machine_usage_vars[m] for m in range(n_machines)) -
            quicksum(setup_costs[m][j] * job_allocation_vars[m, j] for m in range(n_machines) for j in range(n_job_types)) -
            quicksum(maintenance_cost_vars[m] * maintenance_cost_factors[m] for m in range(n_machines)),
            "maximize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_machines': 45,
        'n_job_types': 63,
        'min_processing_cost': 467,
        'max_processing_cost': 527,
        'min_setup_cost': 486,
        'max_setup_cost': 984,
        'min_maintenance_cost': 519,
        'max_maintenance_cost': 2276,
        'er_prob': 0.73,
        'min_n': 225,
        'max_n': 450,
    }

    job_scheduler = ManufacturingJobScheduling(parameters, seed=42)
    instance = job_scheduler.generate_instance()
    solve_status, solve_time, objective_value = job_scheduler.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")