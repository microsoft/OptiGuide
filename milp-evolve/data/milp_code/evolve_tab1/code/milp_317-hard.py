import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

class MaxSatisfiabilityWithTransportation:
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
    
    def generate_random_graph(self, n_households):
        return nx.erdos_renyi_graph(n=n_households, p=self.er_prob, seed=self.seed)

    def generate_households_market_data(self, G):
        for node in G.nodes:
            G.nodes[node]['nutrient_demand'] = np.random.randint(1, 100)

        for u, v in G.edges:
            G[u][v]['transport_cost'] = np.random.randint(1, 20)
            
        return G

    def generate_instances(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_maxsat_graph(n)
        clauses = [(f'v{i},v{j}', 1) for i, j in edges] + [(f'-v{i},-v{j}', 1) for i, j in edges]

        n_households = np.random.randint(self.min_households, self.max_households + 1)
        G = self.generate_random_graph(n_households)
        G = self.generate_households_market_data(G)
        
        market_availability = {node: np.random.randint(1, 40) for node in G.nodes}
        zonal_coordination = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G.edges}

        instance_data = {'clauses': clauses, 'G': G, 'market_availability': market_availability, 
                         'zonal_coordination': zonal_coordination}

        return instance_data

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        G = instance['G']
        market_availability = instance['market_availability']
        zonal_coordination = instance['zonal_coordination']
        
        model = Model("MaxSatisfiabilityWithTransportation")
        var_names = {} 
        
        # Create variables for each literal and clause
        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var
            
        # New variables based on the GDO problem
        household_vars = {f"h{node}": model.addVar(vtype="B", name=f"h{node}") for node in G.nodes}
        market_vars = {f"m{u}_{v}": model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in G.edges}
        zone_vars = {f"z{u}_{v}": model.addVar(vtype="B", name=f"z{u}_{v}") for u, v in G.edges}
        cost_vars = {f"c{node}": model.addVar(vtype="C", name=f"c{node}") for node in G.nodes}
        weekly_budget = model.addVar(vtype="C", name="weekly_budget")
        nutrient_demand_vars = {f"d{node}": model.addVar(vtype="C", name=f"d{node}") for node in G.nodes}
        
        # Objective function - maximize the number of satisfied clauses and minimize costs
        objective_expr = quicksum(
            var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        )

        objective_expr -= quicksum(
            G[u][v]['transport_cost'] * market_vars[f"m{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            market_availability[node] * cost_vars[f"c{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            zonal_coordination[(u, v)] * market_vars[f"m{u}_{v}"]
            for u, v in G.edges
        )
        
        model.setObjective(objective_expr, "maximize")

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
                
        # Add additional constraints based on GDO problem
        for u, v in G.edges:
            model.addCons(
                household_vars[f"h{u}"] + household_vars[f"h{v}"] <= 1,
                name=f"Market_{u}_{v}"
            )

        for u, v in G.edges:
            model.addCons(
                household_vars[f"h{u}"] + household_vars[f"h{v}"] <= 1 + zone_vars[f"z{u}_{v}"],
                name=f"Zone1_{u}_{v}"
            )
            model.addCons(
                household_vars[f"h{u}"] + household_vars[f"h{v}"] >= 2 * zone_vars[f"z{u}_{v}"],
                name=f"Zone2_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                cost_vars[f"c{node}"] + nutrient_demand_vars[f"d{node}"] == 0,
                name=f"Household_{node}_Cost"
            )

        for u, v in G.edges:
            model.addCons(
                G[u][v]['transport_cost'] * zonal_coordination[(u, v)] * market_vars[f"m{u}_{v}"] <= weekly_budget,
                name=f"Budget_{u}_{v}"
            )

        model.addCons(
            weekly_budget <= self.weekly_budget_limit,
            name="Weekly_budget_limit"
        )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 75,
        'max_n': 186,
        'er_prob': 0.24,
        'edge_addition_prob': 0.31,
        'min_households': 195,
        'max_households': 680,
        'weekly_budget_limit': 2529,
    }

    maxsat_gdo = MaxSatisfiabilityWithTransportation(parameters, seed=seed)
    instance = maxsat_gdo.generate_instances()
    solve_status, solve_time = maxsat_gdo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")