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
        # Introducing weights and conditional probabilities from the second MILP
        weights = np.random.randint(1, 10, size=len(clauses))  # Weights for clauses
        conditional_prob = np.random.rand(len(clauses))  # Conditional probabilities for clauses
        
        # Data for the new constraints, incorporating depots and customers
        depot_costs = np.random.randint(self.min_depot_cost, self.max_depot_cost + 1, self.depot_count)
        customer_costs = np.random.randint(self.min_customer_cost, self.max_customer_cost + 1, (self.depot_count, self.customer_count))
        capacities = np.random.randint(self.min_depot_capacity, self.max_depot_capacity + 1, self.depot_count)
        demands = np.random.randint(1, 10, self.customer_count)
        
        res.update({
            'weights': weights,
            'conditional_prob': conditional_prob,
            'depot_costs': depot_costs,
            'customer_costs': customer_costs,
            'capacities': capacities,
            'demands': demands,
        })
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        weights = instance['weights']
        conditional_prob = instance['conditional_prob']
        depot_costs = instance['depot_costs']
        customer_costs = instance['customer_costs']
        capacities = instance['capacities']
        demands = instance['demands']

        model = Model("MaxSatisfiability")
        var_names = {}  # Dictionary to keep track of all variables
        
        # Decision variables for depot optimization
        depot_open = {c: model.addVar(vtype="B", name=f"DepotOpen_{c}") for c in range(self.depot_count)}
        customer_served = {(c, r): model.addVar(vtype="B", name=f"Depot_{c}_Customer_{r}") for c in range(self.depot_count) for r in range(self.customer_count)}
        
        # Create variables for each literal and clause
        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var

        # Objective function - maximize the weighted number of satisfied clauses and minimize total depot and customer service costs
        objective_expr = quicksum(
            weights[idx] * var_names[f"cl_{idx}"] for idx in range(len(clauses))
        ) - quicksum(depot_costs[c] * depot_open[c] for c in range(self.depot_count)) - quicksum(customer_costs[c, r] * customer_served[c, r] for c in range(self.depot_count) for r in range(self.customer_count))

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

            # Adding conditional constraints based on conditional_prob
            condition_var = model.addVar(vtype="B", name=f"cond_{idx}")
            prob = conditional_prob[idx]
            model.addCons(condition_var <= prob, name=f"condition_{idx}")
            model.addCons(clause_var >= condition_var, name=f"conditional_satisfaction_{idx}")

        ### Add new constraints for depot and customer service ###
        # Constraints: Each customer is served by at least one depot
        for r in range(self.customer_count):
            model.addCons(quicksum(customer_served[c, r] for c in range(self.depot_count)) >= 1, f"Customer_{r}_Service")
        
        # Constraints: Only open depots can serve customers
        for c in range(self.depot_count):
            for r in range(self.customer_count):
                model.addCons(customer_served[c, r] <= depot_open[c], f"Depot_{c}_Serve_{r}")
        
        # Constraints: Depots cannot exceed their capacity
        for c in range(self.depot_count):
            model.addCons(quicksum(demands[r] * customer_served[c, r] for r in range(self.customer_count)) <= capacities[c], f"Depot_{c}_Capacity")
        
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
        'depot_count': 10,
        'customer_count': 50,
        'min_depot_cost': 100,
        'max_depot_cost': 1000,
        'min_customer_cost': 100,
        'max_customer_cost': 1000,
        'min_depot_capacity': 100,
        'max_depot_capacity': 1000,
    }
    
    parameters.update({
        'clause_min_weight': 1,
        'clause_max_weight': 10,
        'conditional_prob_min': 0.1,
        'conditional_prob_max': 1.0,
    })

    maxsat = MaxSatisfiability(parameters, seed=seed)
    instance = maxsat.generate_instances()
    solve_status, solve_time = maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")