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

        # Generate flow capacities and initial flows
        edge_flows = {(i, j): {'capacity': np.random.randint(1, 10), 'initial_flow': np.random.random()} for i, j in edges}

        res = {'clauses': clauses, 'edges': edges, 'edge_flows': edge_flows}

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        edges = instance['edges']
        edge_flows = instance['edge_flows']

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

        flow_vars = {f"flow_{i}_{j}": model.addVar(lb=0, ub=edge_flows[(i, j)]['capacity'], name=f"flow_{i}_{j}") for i, j in edges}

        # Objective function - maximize the number of satisfied clauses and the flow sum
        objective_expr = quicksum(var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf)
        flow_obj = quicksum(flow_var for flow_var in flow_vars.values())

        # Combine objectives by giving equal weight to both
        model.setObjective(objective_expr + flow_obj, "maximize")

        # Add constraints for each clause
        for idx, (clause, weight) in enumerate(clauses):
            vars_in_clause = clause.split(',')
            clause_var = var_names[f"cl_{idx}"]

            positive_part = quicksum(var_names[var] for var in vars_in_clause if not var.startswith('-'))
            negative_part = quicksum(1 - var_names[var[1:]] for var in vars_in_clause if var.startswith('-'))

            total_satisfied = positive_part + negative_part

            if weight < np.inf:
                model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")
            else:
                model.addCons(total_satisfied >= 1, name=f"clause_{idx}")

        # Flow conservation constraints
        for node in set([i for i, j in edges] + [j for i, j in edges]):
            incoming_flow = quicksum(flow_vars[f"flow_{i}_{node}"] for i, j in edges if j == node)
            outgoing_flow = quicksum(flow_vars[f"flow_{node}_{j}"] for i, j in edges if i == node)
            model.addCons(incoming_flow == outgoing_flow, name=f"flow_conservation_{node}")

        # Capacity constraints are already handled by the variable bounds

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 56,
        'max_n': 62,
        'er_prob': 0.73,
        'edge_addition_prob': 0.66,
    }
    maxsat = MaxSatisfiability(parameters, seed=seed)
    instance = maxsat.generate_instances()
    solve_status, solve_time = maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")