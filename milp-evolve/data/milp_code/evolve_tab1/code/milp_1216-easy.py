import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SetCoverWithUtility:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        # Set Cover data generation
        nnzrs = int(self.n_rows * self.n_cols * self.density)

        indices = np.random.choice(self.n_cols, size=nnzrs)
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)
        _, col_nrows = np.unique(indices, return_counts=True)

        indices[:self.n_rows] = np.random.permutation(self.n_rows)
        i = 0
        indptr = [0]
        for n in col_nrows:
            if i >= self.n_rows:
                indices[i:i+n] = np.random.choice(self.n_rows, size=n, replace=False)
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_rows, replace=False)
            i += n
            indptr.append(i)

        c = np.random.randint(self.max_coef, size=self.n_cols) + 1
        A = scipy.sparse.csc_matrix(
                (np.ones(len(indices), dtype=int), indices, indptr),
                shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        # Additional Data for Utility Constraints from GISP
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        zone_limits = self.generate_zone_data(G)
        air_quality_limits = self.generate_air_quality_limits(G)
        node_capacities = {node: np.random.randint(self.min_capacity, self.max_capacity) for node in G.nodes}
        
        return { 'c': c, 'indptr_csr': indptr_csr, 'indices_csr': indices_csr,
                 'G': G, 'node_capacities': node_capacities, 'zone_limits': zone_limits,
                 'air_quality_limits': air_quality_limits }

    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.ba_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['revenue'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['cost'] = 1.0

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_zone_data(self, G):
        return {node: np.random.randint(self.min_zone_limit, self.max_zone_limit) for node in G.nodes}

    def generate_air_quality_limits(self, G):
        return {node: np.random.randint(self.min_air_quality, self.max_air_quality) for node in G.nodes}

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        G = instance['G']
        node_capacities = instance['node_capacities']
        zone_limits = instance['zone_limits']
        air_quality_limits = instance['air_quality_limits']

        model = Model("SetCoverWithUtility")
        var_names = {}
        node_vars = {node: model.addVar(vtype="B", name=f"x_{node}") for node in G.nodes}

        # SCP Variables and Objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_scp_{j}", obj=c[j])

        # Add coverage constraints for SCP
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")

        # Utility Variables and Objective
        production_yield = {node: model.addVar(vtype="C", name=f"Y_{node}", lb=0) for node in G.nodes}
        edge_vars = {f"y_{u}_{v}": model.addVar(vtype="B", name=f"y_{u}_{v}") for u, v in G.edges}

        utility_expr = quicksum(G.nodes[node]['revenue'] * node_vars[node] for node in G.nodes) - \
                       quicksum(edge_vars[f"y_{u}_{v}"] for u, v in G.edges)

        # Constraints for Utility
        for u, v in G.edges:
            model.addCons(node_vars[u] + node_vars[v] - edge_vars[f"y_{u}_{v}"] <= 1, name=f"C_{u}_{v}")

        for node in G.nodes:
            model.addCons(production_yield[node] <= zone_limits[node], name=f"Zone_Limit_{node}")

        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            model.addCons(quicksum(production_yield[neighbor] for neighbor in neighbors) <= air_quality_limits[node], name=f"Air_Quality_{node}")

        for node in G.nodes:
            model.addCons(quicksum(edge_vars[f"y_{u}_{v}"] for u, v in G.edges if u == node or v == node) <= node_capacities[node], name=f"Knapsack_Node_{node}")

        # Piecewise Linear Utility Function for diminishing returns
        breakpoint1 = 0.5
        breakpoint2 = 1.0
        penalty_nodes = {node: model.addVar(vtype="C", name=f"penalty_{node}", lb=0) for node in G.nodes}
        for node in G.nodes:
            model.addCons(production_yield[node] <= node_vars[node]*breakpoint1 + (breakpoint2 - breakpoint1)*penalty_nodes[node], name=f"Diminishing_Yield_{node}")

        # Penalty on Air Quality Exceedance
        max_air_quality_penalty = 1000  # Just a random large number for penalty
        penalty_air_quality = {}
        for node in G.nodes:
            if node not in penalty_air_quality:
                penalty_air_quality[node] = model.addVar(vtype="C", name=f"penalty_air_{node}", lb=0)
            neighbors = list(G.neighbors(node))
            model.addCons(quicksum(production_yield[neighbor] for neighbor in neighbors) - penalty_air_quality[node] <= air_quality_limits[node], name=f"Penalty_Air_{node}")

        # Piecewise Linear Cost Function for node capacity penalties
        capacity_penalty = {node: model.addVar(vtype="C", name=f"capacity_penalty_{node}", lb=0) for node in G.nodes}
        for node in G.nodes:
            model.addCons(quicksum(edge_vars[f"y_{u}_{v}"] for u, v in G.edges if u == node or v == node) 
                          <= node_capacities[node] + capacity_penalty[node], name=f"Penalty_Capacity_{node}")

        # Combined Objective
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols)) + utility_expr
        objective_expr += quicksum(penalty_nodes[node] for node in G.nodes)
        objective_expr += quicksum(penalty_air_quality[node] for node in G.nodes)
        objective_expr += quicksum(capacity_penalty[node] for node in G.nodes)

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 750,
        'n_cols': 3000,
        'density': 0.1,
        'max_coef': 100,
        'min_n': 9,
        'max_n': 876,
        'alpha': 0.73,
        'ba_m': 150,
        'min_capacity': 675,
        'max_capacity': 2355,
        'min_zone_limit': 2250,
        'max_zone_limit': 3000,
        'min_air_quality': 112,
        'max_air_quality': 700,
    }

    set_cover_with_utility = SetCoverWithUtility(parameters, seed=seed)
    instance = set_cover_with_utility.generate_instance()
    solve_status, solve_time = set_cover_with_utility.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")