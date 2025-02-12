import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FCMCNFWithCliquesAndWDO:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_commodities(self, G):
        commodities = []
        for k in range(self.n_commodities):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            # integer demands
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)
        
        cliques = list(nx.find_cliques(G.to_undirected()))
        cliques = [clq for clq in cliques if len(clq) >= 3]  # Considering only cliques of size >= 3

        # Generate additional data for WDO
        self.num_neighborhoods = np.random.randint(self.min_neighborhoods, self.max_neighborhoods)
        self.horizon = self.horizon
        self.maintenance_slots = self.maintenance_slots
        neighborhoods = list(range(self.num_neighborhoods))
        machines = list(range(self.num_machines))
        tasks = list(range(self.num_tasks))

        maintenance_schedule = {
            machine: sorted(random.sample(range(self.horizon), self.maintenance_slots))
            for machine in machines
        }

        demand = {machine: random.randint(5, 15) for machine in machines}
        energy_consumption = {machine: random.uniform(1.0, 5.0) for machine in machines}
        
        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'cliques': cliques,
            'machines': machines,
            'tasks': tasks,
            'maintenance_schedule': maintenance_schedule,
            'demand': demand,
            'energy_consumption': energy_consumption,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        cliques = instance['cliques']
        machines = instance['machines']
        tasks = instance['tasks']
        maintenance_schedule = instance['maintenance_schedule']
        demand = instance['demand']
        energy_consumption = instance['energy_consumption']

        model = Model("FCMCNFWithCliquesAndWDO")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}

        machine_assignment_vars = {(machine, task): model.addVar(vtype="B", name=f"machine_{machine}_{task}") for machine in machines for task in tasks}
        maintenance_vars = {(machine, time): model.addVar(vtype="B", name=f"maintenance_{machine}_{time}") for machine in machines for time in range(self.horizon)}
        labor_assignment_vars = {(machine, time): model.addVar(vtype="B", name=f"labor_{machine}_{time}") for machine in machines for time in range(self.horizon)}

        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )

        objective_expr -= quicksum(
            energy_consumption[machine] * machine_assignment_vars[(machine, task)]
            for machine in machines for task in tasks
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        # Adding clique inequalities
        for clq in cliques:
            edges_in_clique = [(u, v) for u in clq for v in clq if (u, v) in edge_list]
            if edges_in_clique:
                clique_expr = quicksum(y_vars[f"y_{u+1}_{v+1}"] for (u, v) in edges_in_clique)
                model.addCons(clique_expr <= len(clq) - 1, f"clique_{'_'.join(map(str, clq))}")

        for machine in machines:
            for t in range(self.horizon):
                if t in maintenance_schedule[machine]:
                    model.addCons(maintenance_vars[(machine, t)] == 1, name=f"MaintenanceScheduled_{machine}_{t}")
                else:
                    model.addCons(maintenance_vars[(machine, t)] == 0, name=f"MaintenanceNotScheduled_{machine}_{t}")

        for t in range(self.horizon):
            model.addCons(
                quicksum(labor_assignment_vars[(machine, t)] for machine in machines) <= self.max_labor_hours_per_day,
                name=f"LaborLimit_{t}"
            )

        model.addCons(
            quicksum(energy_consumption[machine] * machine_assignment_vars[(machine, task)]
                     for machine in machines for task in tasks) <= self.max_energy_consumption,
            name="EnergyConsumptionLimit"
        )

        # Logical constraints linking manufacturing tasks and maintenance
        for machine in machines:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        machine_assignment_vars[(machine, task)] <= maintenance_vars[(machine, t)],
                        name=f"LogicalCondition_{machine}_{task}_{t}"
                    )

        # Conditional resource allocation based on task completion and machine maintenance
        for machine in machines:
            for t in range(self.horizon):
                for task in tasks:
                    model.addCons(
                        machine_assignment_vars[(machine, task)] * maintenance_vars[(machine, t)] * demand[machine] <= self.max_resource_allocation,
                        name=f"ResourceAllocation_{machine}_{task}_{t}"
                    )

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 60,
        'max_n_nodes': 90,
        'min_n_commodities': 7,
        'max_n_commodities': 33,
        'c_range': (55, 250),
        'd_range': (50, 500),
        'ratio': 50,
        'k_max': 10,
        'er_prob': 0.31,
        'min_neighborhoods': 75,
        'max_neighborhoods': 450,
        'num_segments': 10,
        'num_machines': 20,
        'num_tasks': 135,
        'horizon': 18,
        'maintenance_slots': 6,
        'max_labor_hours_per_day': 400,
        'max_energy_consumption': 1000,
        'max_resource_allocation': 200,  
    }

    fcmcnf_cliques_wdo = FCMCNFWithCliquesAndWDO(parameters, seed=seed)
    instance = fcmcnf_cliques_wdo.generate_instance()
    solve_status, solve_time = fcmcnf_cliques_wdo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")