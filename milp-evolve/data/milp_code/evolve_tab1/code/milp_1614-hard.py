import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DisasterReliefOptimization:
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
            G.nodes[zone]['urgency'] = np.random.uniform(0.5, 1.5)  # Urgency level
            G.nodes[zone]['infrastructure'] = np.random.uniform(0.5, 1.5)  # Infrastructure level

        for u, v in G.edges:
            G[u][v]['cost'] = (G.nodes[u]['aid_value'] + G.nodes[v]['aid_value']) / float(self.cost_param)
            G[u][v]['transport_availability'] = np.random.uniform(0.5, 1.5)  # Transport resource availability

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_aids_costs(G)
        res = {'G': G}

        for zone in G.nodes:
            res[f'aid_cost_fixed_{zone}'] = np.random.randint(100, 200)
            res[f'aid_cost_variable_{zone}'] = np.random.uniform(2, 10)

        for u, v in G.edges:
            break_points = np.linspace(0, G[u][v]['cost'], self.piece_count + 1)
            slopes = np.diff(break_points)
            res[f'break_points_{u}_{v}'] = break_points
            res[f'slopes_{u}_{v}'] = slopes

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        
        model = Model("DisasterReliefOptimization")
        zone_vars = {f"z{zone}":  model.addVar(vtype="B", name=f"z{zone}") for zone in G.nodes}
        edge_vars = {f"e{u}_{v}": model.addVar(vtype="B", name=f"e{u}_{v}") for u, v in G.edges}
        piece_vars = {f"p{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"p{u}_{v}_{k}") for u, v in G.edges for k in range(self.piece_count)}

        # Objective function to maximize aid value while minimizing costs and prioritizing high-urgency zones
        objective_expr = quicksum(
            G.nodes[zone]['aid_value'] * G.nodes[zone]['urgency'] * zone_vars[f"z{zone}"]
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
        
        # Transport availability constraints
        transport_vars = {f"t{u}_{v}": model.addVar(vtype="C", name=f"t{u}_{v}") for u, v in G.edges}
        
        for u, v in G.edges:
            max_transport = G[u][v]['transport_availability']
            model.addCons(transport_vars[f"t{u}_{v}"] <= edge_vars[f'e{u}_{v}'] * max_transport)
            model.addCons(transport_vars[f"t{u}_{v}"] >= edge_vars[f'e{u}_{v}'] * G.nodes[u]['infrastructure'] * G.nodes[v]['infrastructure'])

        # Budget constraints
        for zone in G.nodes:
            model.addCons(
                zone_vars[f"z{zone}"] * instance[f'aid_cost_fixed_{zone}'] + zone_vars[f"z{zone}"] * instance[f'aid_cost_variable_{zone}'] <= self.budget_threshold
            )

        # Set objective and solve the model
        model.setObjective(objective_expr, "maximize")
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_zones': 200,
        'max_zones': 600,
        'er_prob': 0.56,
        'graph_type': 'BA',
        'barabasi_m': 10,
        'cost_param': 3000.0,
        'piece_count': 10,
        'budget_threshold': 10000,
    }

    ealloc = DisasterReliefOptimization(parameters, seed=seed)
    instance = ealloc.generate_instance()
    solve_status, solve_time = ealloc.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")