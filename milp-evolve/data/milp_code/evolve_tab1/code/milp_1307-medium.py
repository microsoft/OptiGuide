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

        # New data generation for flow, weights, and capacities
        node_weights = np.random.randint(1, self.max_weight, n)
        knapsack_capacity = np.random.randint(self.min_capacity, self.max_capacity)
        flow_capacities = {edge: np.random.randint(1, self.max_flow_capacity) for edge in edges}

        # Additional data for demographic constraints and travel costs
        demographic_data = np.random.dirichlet(np.ones(5), size=n).tolist()  # 5 demographic groups
        travel_costs = {edge: np.random.uniform(5.0, 30.0) for edge in edges}

        res = {
            'clauses': clauses,
            'node_weights': node_weights,
            'knapsack_capacity': knapsack_capacity,
            'flow_capacities': flow_capacities,
            'nodes': list(range(n)),
            'edges': edges,
            'demographic_data': demographic_data,
            'travel_costs': travel_costs,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        node_weights = instance['node_weights']
        knapsack_capacity = instance['knapsack_capacity']
        flow_capacities = instance['flow_capacities']
        nodes = instance['nodes']
        edges = instance['edges']
        demographic_data = instance['demographic_data']
        travel_costs = instance['travel_costs']

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
        
        # New flow variables for each edge
        flow_vars = {f"Flow_{i}_{j}": model.addVar(vtype="C", name=f"Flow_{i}_{j}", lb=0) for i, j in edges}

        # New demographic variables
        demographic_vars = {(g, i): model.addVar(vtype="B", name=f"Demographic_{g}_{i}") for g in range(5) for i in range(len(nodes))}

        # Objective function - maximize the number of satisfied clauses
        objective_expr = quicksum(
            var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
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

        # Add knapsack constraint on literals' weights
        knapsack_constraint = quicksum(node_weights[int(lit[1:])] * var_names[lit] for lit in var_names if lit.startswith('v'))
        model.addCons(knapsack_constraint <= knapsack_capacity, name="knapsack")

        # Create flow variables and add flow constraints
        flow = {}
        for (i, j) in edges:
            flow[i, j] = model.addVar(vtype="C", name=f"flow_{i}_{j}")
            flow[j, i] = model.addVar(vtype="C", name=f"flow_{j}_{i}")
            model.addCons(flow[i, j] <= flow_capacities[(i, j)], name=f"flow_capacity_{i}_{j}")
            model.addCons(flow[j, i] <= flow_capacities[(i, j)], name=f"flow_capacity_{j}_{i}")

        # Flow conservation constraints
        for node in nodes:
            model.addCons(
                quicksum(flow[i, j] for (i, j) in edges if j == node) ==
                quicksum(flow[i, j] for (i, j) in edges if i == node),
                name=f"flow_conservation_{node}"
            )

        # Implementing Big M Formulation for novel constraints
        M = self.max_weight  # Big M constraint
        for (i, j) in edges:
            y = model.addVar(vtype="B", name=f"y_{i}_{j}")  # auxiliary binary variable
            model.addCons(var_names[f'v{i}'] + var_names[f'v{j}'] - 2 * y <= 0, name=f"bigM1_{i}_{j}")
            model.addCons(var_names[f'v{i}'] + var_names[f'v{j}'] + M * (y - 1) >= 0, name=f"bigM2_{i}_{j}")

        # Additional constraints to capture travel costs and demographic balance
        for (g, i) in demographic_vars:
            model.addCons(demographic_vars[(g, i)] <= demographic_data[i][g], name=f"Demographic_balance_{g}_{i}")

        for (i, j) in travel_costs:
            model.addCons(flow_vars[f"Flow_{i}_{j}"] <= flow_capacities[(i, j)], name=f"Flow_cover_{i}_{j}")

        # Modifying the objective to include travel costs and demographic balance
        objective_expr -= quicksum(travel_costs[(i, j)] * flow_vars[f"Flow_{i}_{j}"] for (i, j) in edges)
        objective_expr += quicksum(self.demographic_weights[g] * demographic_vars[(g, i)] for g in range(5) for i in range(len(nodes)))

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 75,
        'max_n': 375,
        'er_prob': 0.66,
        'edge_addition_prob': 0.52,
        'max_weight': 800,
        'min_capacity': 1200,
        'max_capacity': 2000,
        'max_flow_capacity': 1800,
        'demographic_weights': (10.0, 15.0, 20.0, 25.0, 30.0),
    }

    maxsat = MaxSatisfiability(parameters, seed=seed)
    instance = maxsat.generate_instances()
    solve_status, solve_time = maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")