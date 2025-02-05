import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_graph(self):
        n_zones = np.random.randint(self.min_zones, self.max_zones)
        if self.graph_type == 'ER':
            G = nx.erdos_renyi_graph(n=n_zones, p=self.er_prob, seed=self.seed)
        elif self.graph_type == 'BA':
            G = nx.barabasi_albert_graph(n=n_zones, m=self.barabasi_m, seed=self.seed)
        return G

    def generate_aids_costs(self, G):
        for zone in G.nodes:
            G.nodes[zone]['aid_value'] = np.random.randint(50, 150)
            G.nodes[zone]['penalty'] = np.random.randint(1, 50)

        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['aid_value'] + G.nodes[v]['aid_value']) / float(self.cost_param)

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_aids_costs(G)
        res = {'G': G}
        
        for u, v in G.edges:
            break_points = np.linspace(0, G[u][v]['cost'], self.piece_count + 1)
            slopes = np.diff(break_points)
            res[f'break_points_{u}_{v}'] = break_points
            res[f'slopes_{u}_{v}'] = slopes
        
        for zone in G.nodes:
            res[f'aid_cost_fixed_{zone}'] = np.random.randint(100, 200)
            res[f'aid_cost_variable_{zone}'] = np.random.uniform(2, 10)

        ### additional data generation for the new constraints and objective
        for u, v in G.edges:
            res[f'max_aid_cost_{u}_{v}'] = np.random.uniform(100, 200)
            res[f'min_aid_cost_{u}_{v}'] = np.random.uniform(50, 100)
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        
        model = Model("EmergencyResourceAllocation")
        zone_vars = {f"z{zone}":  model.addVar(vtype="B", name=f"z{zone}") for zone in G.nodes}
        edge_vars = {f"e{u}_{v}": model.addVar(vtype="B", name=f"e{u}_{v}") for u, v in G.edges}
        piece_vars = {f"p{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"p{u}_{v}_{k}") for u, v in G.edges for k in range(self.piece_count)}

        # Modified objective function to maximize aid value while considering costs
        objective_expr = quicksum(
            G.nodes[zone]['aid_value'] * zone_vars[f"z{zone}"]
            for zone in G.nodes
        ) - quicksum(
            instance[f'slopes_{u}_{v}'][k] * piece_vars[f"p{u}_{v}_{k}"]
            for u, v in G.edges
            for k in range(self.piece_count)
        )

        # Applying Piecewise Linear Function Constraints
        M = 10000  # Big M constant, should be large enough
        for u, v in G.edges:
            model.addCons(
                zone_vars[f"z{u}"] + zone_vars[f"z{v}"] - edge_vars[f"e{u}_{v}"] * M <= 1,
                name=f"C_{u}_{v}"
            )
                
        for u, v in G.edges:
            model.addCons(quicksum(piece_vars[f"p{u}_{v}_{k}"] for k in range(self.piece_count)) == edge_vars[f'e{u}_{v}'])
            
            for k in range(self.piece_count):
                model.addCons(piece_vars[f"p{u}_{v}_{k}"] <= instance[f'break_points_{u}_{v}'][k+1] - instance[f'break_points_{u}_{v}'][k] + (1 - edge_vars[f'e{u}_{v}']) * M)

        for zone in G.nodes:
            model.addCons(
                zone_vars[f"z{zone}"] * instance[f'aid_cost_fixed_{zone}'] + zone_vars[f"z{zone}"] * instance[f'aid_cost_variable_{zone}'] <= self.budget_threshold
            )

        ### Define semi-continuous variables for expenses
        expense_vars = {f"s{u}_{v}": model.addVar(vtype="C", name=f"s{u}_{v}") for u, v in G.edges}
       
        # Add constraints for semi-continuous variables representing costs
        for u, v in G.edges:
            max_aid_cost = instance[f'max_aid_cost_{u}_{v}']
            min_aid_cost = instance[f'min_aid_cost_{u}_{v}']
            model.addCons(expense_vars[f"s{u}_{v}"] >= edge_vars[f"e{u}_{v}"] * min_aid_cost)
            model.addCons(expense_vars[f"s{u}_{v}"] <= edge_vars[f"e{u}_{v}"] * max_aid_cost)
            objective_expr -= expense_vars[f"s{u}_{v}"]

        ### Set objective and solve the model
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_zones': 81,
        'max_zones': 768,
        'er_prob': 0.45,
        'graph_type': 'BA',
        'barabasi_m': 18,
        'cost_param': 3000.0,
        'piece_count': 15,
        'budget_threshold': 50000,
    }
    params = {
        'max_aid_cost': 200.00,
        'min_aid_cost': 50.00,
    }
    parameters.update(params)

    ealloc = EmergencyResourceAllocation(parameters, seed=seed)
    instance = ealloc.generate_instance()
    solve_status, solve_time = ealloc.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")