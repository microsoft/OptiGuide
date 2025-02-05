import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedSetCoverWithFlow:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_rows * self.n_cols * self.density)

        # compute number of rows per column
        indices = np.random.choice(self.n_cols, size=nnzrs)  # random column indexes
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)  # force at least 2 rows per col
        _, col_nrows = np.unique(indices, return_counts=True)

        # for each column, sample random rows
        indices[:self.n_rows] = np.random.permutation(self.n_rows) # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # empty column, fill with random rows
            if i >= self.n_rows:
                indices[i:i+n] = np.random.choice(self.n_rows, size=n, replace=False)

            # partially filled column, complete with random rows among remaining ones
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_rows, replace=False)

            i += n
            indptr.append(i)

        # objective coefficients
        c = np.random.randint(self.max_coef, size=self.n_cols) + 1

        # generate a random graph to introduce additional complexity
        graph = nx.barabasi_albert_graph(self.n_cols, self.num_edges_per_node, seed=self.seed)
        capacities = np.random.randint(1, self.max_capacity, size=len(graph.edges))
        flows = np.random.uniform(0, self.max_flow, size=len(graph.edges))

        # generate start and end nodes for flow network
        source_node, sink_node = 0, self.n_cols - 1

        # convert graph to adjacency list
        adj_list = {i: [] for i in range(self.n_cols)}
        for idx, (u, v) in enumerate(graph.edges):
            adj_list[u].append((v, flows[idx], capacities[idx]))
            adj_list[v].append((u, flows[idx], capacities[idx])) # for undirected edges

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
                (np.ones(len(indices), dtype=int), indices, indptr),
                shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res =  {'c': c, 
                'indptr_csr': indptr_csr, 
                'indices_csr': indices_csr,
                'adj_list': adj_list,
                'source_node': source_node,
                'sink_node': sink_node}

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        adj_list = instance['adj_list']
        source_node = instance['source_node']
        sink_node = instance['sink_node']

        model = Model("EnhancedSetCoverWithFlow")
        var_names = {}

        # Create binary variables and set objective coefficients
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])

        flow_vars = {}
        # Create flow variables for edges
        for u in adj_list:
            for v, _, capacity in adj_list[u]:
                flow_vars[(u, v)] = model.addVar(vtype='C', lb=0, ub=capacity, name=f"f_{u}_{v}")

        # Add constraints to ensure each row is covered
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        # Add flow constraints
        for node in adj_list:
            if node == source_node:
                # flow balance for source node:
                model.addCons(quicksum(flow_vars[(source_node, v)] for v, _, _ in adj_list[source_node]) == 0, f"flow_source_{source_node}")
            elif node == sink_node:
                # flow balance for sink node:
                model.addCons(quicksum(flow_vars[(u, sink_node)] for u, _, _ in adj_list[sink_node]) == 0, f"flow_sink_{sink_node}")
            else:
                inflow = quicksum(flow_vars[(u, node)] for u, _, _ in adj_list[node] if (u, node) in flow_vars)
                outflow = quicksum(flow_vars[(node, v)] for v, _, _ in adj_list[node] if (node, v) in flow_vars)
                model.addCons(inflow - outflow == 0, f"flow_balance_{node}")

        # Composite objective: Minimize total cost and maximize total flow
        cost_term = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        flow_term = quicksum(flow_vars[(u, v)] for u, v in flow_vars)
        objective_expr = cost_term - self.flow_weight * flow_term

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 750,
        'n_cols': 375,
        'density': 0.24,
        'max_coef': 25,
        'num_edges_per_node': 1,
        'max_capacity': 150,
        'max_flow': 2,
        'flow_weight': 0.17,
    }
    
    enhanced_set_cover_with_flow = EnhancedSetCoverWithFlow(parameters, seed=seed)
    instance = enhanced_set_cover_with_flow.generate_instance()
    solve_status, solve_time = enhanced_set_cover_with_flow.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")