import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

class SimplifiedSupplyChainOptimization:
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

        # Generate piecewise linear function parameters
        num_pieces = self.num_pieces
        breakpoints = sorted(np.random.uniform(self.min_resource, self.max_resource, num_pieces - 1))
        slopes = np.random.uniform(self.min_slope, self.max_slope, num_pieces)
        intercepts = np.random.uniform(self.min_intercept, self.max_intercept, num_pieces)

        res.update({
            'breakpoints': breakpoints,
            'slopes': slopes,
            'intercepts': intercepts
        })
        
        # Generate humanitarian aid demand data
        humanitarian_regions = list(np.random.choice(range(self.num_regions), self.num_humanitarian_regions, replace=False))
        commercial_regions = [r for r in range(self.num_regions) if r not in humanitarian_regions]
        
        humanitarian_demand = np.random.uniform(self.min_demand, self.max_demand, self.num_humanitarian_regions)
        commercial_demand = np.random.uniform(self.min_demand, self.max_demand, self.num_regions - self.num_humanitarian_regions)

        res.update({
            'humanitarian_regions': humanitarian_regions,
            'commercial_regions': commercial_regions,
            'humanitarian_demand': humanitarian_demand,
            'commercial_demand': commercial_demand
        })

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        breakpoints = instance['breakpoints']
        slopes = instance['slopes']
        intercepts = instance['intercepts']
        humanitarian_regions = instance['humanitarian_regions']
        commercial_regions = instance['commercial_regions']
        humanitarian_demand = instance['humanitarian_demand']
        commercial_demand = instance['commercial_demand']

        model = Model("SimplifiedSupplyChainOptimization")
        var_names = {}  

        # Create variables for each literal and clause
        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var

        # Create resource variable
        resource = model.addVar(vtype="C", lb=self.min_resource, ub=self.max_resource, name="resource")

        # Piecewise linear cost function
        cost = model.addVar(vtype="C", name="cost")
        for i in range(len(breakpoints) - 1):
            model.addCons(
                cost >= slopes[i] * resource + intercepts[i],
                name=f"cost_piece_{i}")
        
        model.addCons(
            cost >= slopes[-1] * resource + intercepts[-1],
            name="cost_last_piece")
        
        # Create energy consumption variable
        energy = model.addVar(vtype="C", name="E_total")
        
        # Objective function - maximize the number of satisfied clauses minus resource cost and energy
        objective_expr = quicksum(
            var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        ) - cost - energy * self.energy_rate

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

        # Total energy consumption constraint
        model.addCons(energy <= self.total_energy_limit, name="total_energy_limit")
        
        # Demand constraints for humanitarian aid regions
        for idx, region in enumerate(humanitarian_regions):
            var_name = f"demand_h_{region}"
            var_names[var_name] = model.addVar(vtype="C", name=var_name)
            model.addCons(var_names[var_name] >= humanitarian_demand[idx], name=f"demand_h_constraint_{region}")
        
        # Demand constraints for commercial regions
        for idx, region in enumerate(commercial_regions):
            var_name = f"demand_c_{region}"
            var_names[var_name] = model.addVar(vtype="C", name=var_name)
            model.addCons(var_names[var_name] >= commercial_demand[idx], name=f"demand_c_constraint_{region}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 10,
        'max_n': 250,
        'er_prob': 0.45,
        'edge_addition_prob': 0.52,
        'num_pieces': 30,
        'min_resource': 0,
        'max_resource': 100,
        'min_slope': 0,
        'max_slope': 10,
        'min_intercept': 0,
        'max_intercept': 75,
        'energy_rate': 10,
        'total_energy_limit': 5000,
        'num_regions': 3,
        'num_humanitarian_regions': 1,
        'min_demand': 75,
        'max_demand': 100,
    }

    optimizer = SimplifiedSupplyChainOptimization(parameters, seed=seed)
    instance = optimizer.generate_instances()
    solve_status, solve_time = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")