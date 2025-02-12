import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

class MaxSatisfiabilityWithFLP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_bipartite_graph(self, n1, n2, p):
        return bipartite.random_graph(n1, n2, p, seed=self.seed)

    def generate_maxsat_graph(self, n):
        divider = np.random.randint(1, 6)
        G = self.generate_bipartite_graph(n // divider, n - n // divider, self.er_prob)

        n_edges = len(G.edges)
        edges = list(G.edges)

        added_edges = 0
        while added_edges < n_edges * self.edge_addition_prob:
            i, j = np.random.randint(0, n), np.random.randint(0, n)
            if (i, j) not in edges and (j, i) not in edges:
                added_edges += 1
                edges.append((i, j))

        return edges

    def generate_instances(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_maxsat_graph(n)
        clauses = [(f'v{i},v{j}', 1) for i, j in edges] + [(f'-v{i},-v{j}', 1) for i, j in edges]
        
        # Facility location data
        n_facilities = np.random.randint(5, 15)
        n_clients = n
        operating_cost = np.random.randint(5, 20, size=n_facilities).tolist()
        assignment_cost = np.random.randint(1, 10, size=(n_clients, n_facilities)).tolist()
        capacity = np.random.randint(5, 20, size=n_facilities).tolist()

        res = {
            'clauses': clauses,
            'n_facilities': n_facilities,
            'n_clients': n_clients,
            'operating_cost': operating_cost,
            'assignment_cost': assignment_cost,
            'capacity': capacity
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        n_facilities = instance['n_facilities']
        n_clients = instance['n_clients']
        operating_cost = instance['operating_cost']
        assignment_cost = instance['assignment_cost']
        capacity = instance['capacity']

        model = Model("MaxSatisfiabilityWithFLP")
        var_names = {}  
        y_vars = {}
        x_vars = {}

        # Create variables for each literal and clause
        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var
        
        # Create variables for FLP
        for j in range(n_facilities):
            y_vars[j] = model.addVar(vtype="B", name=f"y_{j}")
            for i in range(n_clients):
                x_vars[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Objective function - maximize the number of satisfied clauses and minimize the FLP costs
        objective_expr = quicksum(
            var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        ) - quicksum(operating_cost[j] * y_vars[j] for j in range(n_facilities)) - quicksum(
            assignment_cost[i][j] * x_vars[i, j] for i in range(n_clients) for j in range(n_facilities)
        )

        # Add constraints for each clause
        for idx, (clause, weight) in enumerate(clauses):
            vars_in_clause = clause.split(',')
            clause_var = var_names[f"cl_{idx}"]
            
            # Define the positive and negative parts
            positive_part = quicksum(var_names[var] for var in vars_in_clause if not var.startswith('-'))
            negative_part = quicksum(1 - var_names[var[1:]] for var in vars_in_clause if var.startswith('-'))
            
            # Total satisfied variables in the clause
            total_satisfied = positive_part + negative_part
            
            if weight < np.inf:
                model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")
            else:
                model.addCons(total_satisfied >= 1, name=f"clause_{idx}")

        # Add FLP constraints: Each client assigned to exactly one open facility
        for i in range(n_clients):
            model.addCons(quicksum(x_vars[i, j] for j in range(n_facilities)) == 1, name=f"client_{i}_assignment")
        
        # Facility capacity constraints
        for j in range(n_facilities):
            model.addCons(quicksum(x_vars[i, j] for i in range(n_clients)) <= capacity[j] * y_vars[j], name=f"facility_{j}_capacity")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
            
if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 37,
        'max_n': 875,
        'er_prob': 0.31,
        'edge_addition_prob': 0.38,
    }

    maxsat_flp = MaxSatisfiabilityWithFLP(parameters, seed=seed)
    instance = maxsat_flp.generate_instances()
    solve_status, solve_time = maxsat_flp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")