import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FactoryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_machines * self.n_tasks * self.density)
        indices = np.random.choice(self.n_tasks, size=nnzrs)  
        indices[:2 * self.n_tasks] = np.repeat(np.arange(self.n_tasks), 2)  
        
        _, col_nrows = np.unique(indices, return_counts=True)
        indices[:self.n_machines] = np.random.permutation(self.n_machines) 
        i = 0
        indptr = [0]
        for n in col_nrows:
            if i >= self.n_machines:
                indices[i:i + n] = np.random.choice(self.n_machines, size=n, replace=False)
            elif i + n > self.n_machines:
                remaining_rows = np.setdiff1d(np.arange(self.n_machines), indices[i:self.n_machines], assume_unique=True)
                indices[self.n_machines:i + n] = np.random.choice(remaining_rows, size=i + n - self.n_machines, replace=False)
            i += n
            indptr.append(i)

        c = np.random.randint(self.max_coef, size=self.n_tasks) + 1

        graph = nx.barabasi_albert_graph(self.n_tasks, self.num_edges_per_task, seed=self.seed)
        capacities = np.random.randint(1, self.max_capacity, size=len(graph.edges))
        setup_times = np.random.uniform(0, self.max_setup_time, size=len(graph.edges))

        source_node, sink_node = 0, self.n_tasks - 1

        adj_list = {i: [] for i in range(self.n_tasks)}
        for idx, (u, v) in enumerate(graph.edges):
            adj_list[u].append((v, setup_times[idx], capacities[idx]))
            adj_list[v].append((u, setup_times[idx], capacities[idx]))  

        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_machines, self.n_tasks)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {'c': c,
               'indptr_csr': indptr_csr,
               'indices_csr': indices_csr,
               'adj_list': adj_list,
               'source_node': source_node,
               'sink_node': sink_node}

        task_times = {i: (np.random.randint(0, self.max_start_time // 2), 
                            np.random.randint(self.max_start_time //2, self.max_end_time)) 
                        for i in range(self.n_tasks)}
        processing_times = {(u, v): np.random.randint(1, self.max_processing_time) for u, v in graph.edges}
        hazards = {i: np.random.normal(0, self.hazard_stddev, size=2) for i in range(self.n_tasks)}

        res.update({'task_times': task_times, 'processing_times': processing_times, 'hazards': hazards})

        Big_M_list = np.random.randint(100, 500, size=self.n_tasks)
        res.update({'Big_M_list': Big_M_list}) 

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        adj_list = instance['adj_list']
        source_node = instance['source_node']
        sink_node = instance['sink_node']
        task_times = instance['task_times']
        hazards = instance['hazards']
        Big_M_list = instance['Big_M_list']

        model = Model("FactoryOptimization")
        machine_vars = {}

        for j in range(self.n_tasks):
            machine_vars[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        setup_vars = {}
        for u in adj_list:
            for v, _, capacity in adj_list[u]:
                setup_vars[(u, v)] = model.addVar(vtype='C', lb=0, ub=capacity, name=f"s_{u}_{v}")

        for row in range(self.n_machines):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(machine_vars[j] for j in cols) >= 1, f"machine_setup_{row}")

        for task in adj_list:
            if task == source_node:
                model.addCons(quicksum(setup_vars[(source_node, v)] for v, _, _ in adj_list[source_node]) >= 1,
                              f"setup_source_{source_node}")
            elif task == sink_node:
                model.addCons(quicksum(setup_vars[(u, sink_node)] for u, _, _ in adj_list[sink_node]) >= 1,
                              f"setup_sink_{sink_node}")
            else:
                inflow = quicksum(setup_vars[(u, task)] for u, _, _ in adj_list[task] if (u, task) in setup_vars)
                outflow = quicksum(setup_vars[(task, v)] for v, _, _ in adj_list[task] if (task, v) in setup_vars)
                model.addCons(inflow - outflow == 0, f"setup_balance_{task}")

        task_vars = {}
        early_penalty_vars = {}
        late_penalty_vars = {}
        hazard_activation_vars = {}

        for j in range(self.n_tasks):
            task_vars[j] = model.addVar(vtype='C', name=f"t_{j}")
            early_penalty_vars[j] = model.addVar(vtype='C', name=f"e_{j}")
            late_penalty_vars[j] = model.addVar(vtype='C', name=f"l_{j}")
            hazard_activation_vars[j] = model.addVar(vtype='B', name=f"hazard_{j}")

            task_start, task_end = task_times[j]
            hazard_start, hazard_end = hazards[j]
            Big_M = Big_M_list[j]

            model.addCons(task_vars[j] >= (task_start + hazard_start) - Big_M * (1 - hazard_activation_vars[j]), 
                          f"task_window_start_M_{j}")
            model.addCons(task_vars[j] <= (task_end + hazard_end) + Big_M * (1 - hazard_activation_vars[j]), 
                          f"task_window_end_M_{j}")

            model.addCons(early_penalty_vars[j] >= (task_start + hazard_start) - task_vars[j], 
                          f"early_penalty_{j}")
            model.addCons(late_penalty_vars[j] >= task_vars[j] - (task_end + hazard_end), 
                          f"late_penalty_{j}")

        cost_term = quicksum(machine_vars[j] * c[j] for j in range(self.n_tasks))
        setup_term = quicksum(setup_vars[(u, v)] for u, v in setup_vars)
        hazard_penalty_term = quicksum(early_penalty_vars[j] + late_penalty_vars[j] for j in range(self.n_tasks))

        objective_expr = cost_term - self.setup_weight * setup_term + self.hazard_penalty_weight * hazard_penalty_term

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_machines': 3000,
        'n_tasks': 140,
        'density': 0.17,
        'max_coef': 2250,
        'num_edges_per_task': 75,
        'max_capacity': 600,
        'max_setup_time': 50,
        'setup_weight': 0.73,
        'max_start_time': 480,
        'max_end_time': 2880,
        'max_processing_time': 540,
        'hazard_penalty_weight': 0.8,
        'hazard_stddev': 150,
    }

    factory_optimization = FactoryOptimization(parameters, seed=seed)
    instance = factory_optimization.generate_instance()
    solve_status, solve_time = factory_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")