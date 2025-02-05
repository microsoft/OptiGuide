import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

class MaxSatisfiability:
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

        res = {'clauses': clauses}
        ### new instance data code starts here
        weights = np.random.randint(1, 10, size=len(clauses))  # Generate weights for clauses
        conditional_prob = np.random.rand(len(clauses))  # Generate conditional probabilities for clauses
        res.update({'weights': weights, 'conditional_prob': conditional_prob})
        ### new instance data code ends here
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        weights = instance['weights']
        conditional_prob = instance['conditional_prob']

        model = Model("MaxSatisfiability")
        var_names = {}  

        # Create variables for each literal and clause
        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var

        # Objective function - maximize the weighted number of satisfied clauses
        objective_expr = quicksum(
            weights[idx] * var_names[f"cl_{idx}"] for idx in range(len(clauses))
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
            
            model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")

            ### new constraints and variables and objective code starts here
            # Adding conditional constraints based on conditional_prob
            condition_var = model.addVar(vtype="B", name=f"cond_{idx}")
            prob = conditional_prob[idx]
            model.addCons(condition_var <= prob, name=f"condition_{idx}")
            model.addCons(clause_var >= condition_var, name=f"conditional_satisfaction_{idx}")
            ### new constraints and variables and objective code ends here

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
            
if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 54,
        'max_n': 562,
        'er_prob': 0.31,
        'edge_addition_prob': 0.24,
    }
    ### new parameter code starts here:
    parameters.update({
        'clause_min_weight': 1,
        'clause_max_weight': 10,
        'conditional_prob_min': 0.1,
        'conditional_prob_max': 1.0,
    })
    ### new parameter code ends here

    maxsat = MaxSatisfiability(parameters, seed=seed)
    instance = maxsat.generate_instances()
    solve_status, solve_time = maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")