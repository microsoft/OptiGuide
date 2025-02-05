import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class LogisticsHubOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.N_min_n, self.N_max_n)
        if self.graph_type == 'ER':
            G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        elif self.graph_type == 'BA':
            G = nx.barabasi_albert_graph(n=n_nodes, m=self.barabasi_m, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['node_revenue'] = np.random.randint(1, 100)
            G.nodes[node]['edge_flow_penalty'] = np.random.uniform(1.0, 10.0)
            G.nodes[node]['max_capacity'] = np.random.randint(100, 500)

        for u, v in G.edges:
            G[u][v]['variable_cost_metric'] = (G.nodes[u]['node_revenue'] + G.nodes[v]['node_revenue']) / float(self.cost_metric_param)
            G[u][v]['flow_penalty'] = np.random.uniform(0.5, 2.0)

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = {(u, v) for u, v in G.edges if np.random.random() <= self.alpha}
        res = {'G': G, 'E2': E2}

        for node in G.nodes:
            res[f'handling_cost_{node}'] = np.random.randint(300, 500)
            res[f'operating_cost_{node}'] = np.random.uniform(1, 5)
            res[f'market_utilization_{node}'] = np.random.uniform(0.1, 0.9)

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, E2 = instance['G'], instance['E2']

        model = Model("LogisticsHubOptimization")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piece_vars = {f"t{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"t{u}_{v}_{k}") for u, v in G.edges for k in range(self.num_pieces)}
        handling_vars = {f"h{node}": model.addVar(vtype="I", lb=0, name=f"h{node}") for node in G.nodes}
        capacity_vars = {f"cap{node}": model.addVar(vtype="C", lb=0, name=f"cap{node}") for node in G.nodes}
        utilization_vars = {f"util{node}": model.addVar(vtype="C", lb=0, name=f"util{node}") for node in G.nodes}
        extra_vars = {f"z{node}": model.addVar(vtype="I", lb=0, ub=20, name=f"z{node}") for node in G.nodes}

        objective_expr = quicksum(
            G.nodes[node]['node_revenue'] * node_vars[f"x{node}"] - instance[f'handling_cost_{node}'] * handling_vars[f"h{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['variable_cost_metric'] * piece_vars[f"t{u}_{v}_{k}"]
            for u, v in E2
            for k in range(self.num_pieces)
        )
        
        objective_expr += quicksum(
            G[u][v]['flow_penalty'] * edge_vars[f'y{u}_{v}']
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            instance[f'operating_cost_{node}'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        # Applying Piecewise Linear Function Constraints
        M = 10000  # Big M constant, should be large enough
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] * M <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] * M <= M,
                    name=f"C_{u}_{v}"
                )
                
        for u, v in E2:
            model.addCons(quicksum(piece_vars[f"t{u}_{v}_{k}"] for k in range(self.num_pieces)) == edge_vars[f'y{u}_{v}'])
            
            for k in range(self.num_pieces):
                model.addCons(piece_vars[f"t{u}_{v}_{k}"] <= (G[u][v]['variable_cost_metric'] / (k+1)) + (1 - edge_vars[f'y{u}_{v}']) * M)

        for node in G.nodes:
            # New handling volume constraint tied to node capacity
            model.addCons(handling_vars[f"h{node}"] <= G.nodes[node]['max_capacity'] * node_vars[f"x{node}"], name=f"NewVolumeConstraint_{node}")

            # Market utilization constraint
            model.addCons(utilization_vars[f"util{node}"] >= instance[f'market_utilization_{node}'] * node_vars[f"x{node}"], name=f"HandlingCapacityConstraint_{node}")
            
            # Penalty for exceeding distance
            model.addCons(extra_vars[f"z{node}"] >= G.nodes[node]['edge_flow_penalty'] * (1 - node_vars[f"x{node}"]), name=f"HandlingPenaltyConstraint_{node}")

        # Budget constraint: Handling Costs + Operating Costs
        total_budget = quicksum(instance[f'handling_cost_{node}'] * handling_vars[f"h{node}"] +
                                instance[f'operating_cost_{node}'] * node_vars[f"x{node}"]
                                for node in G.nodes)
        model.addCons(total_budget <= self.transport_budget, name="HubBudgetConstraint")

        # Adding the handling costs constraint
        handling_costs = quicksum((handling_vars[f"h{node}"] - node_vars[f"x{node}"]) * G.nodes[node]['node_revenue'] for node in G.nodes)
        model.addCons(handling_costs <= self.total_handling_capacity, name="TransportCostConstraint")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'N_min_n': 40,
        'N_max_n': 1000,
        'er_prob': 0.53,
        'graph_type': 'ER',
        'barabasi_m': 20,
        'cost_metric_param': 2000.0,
        'alpha': 0.44,
        'num_pieces': 80,
        'transport_budget': 100000,
        'total_handling_capacity': 10000,
    }

    logistics = LogisticsHubOptimization(parameters, seed=seed)
    instance = logistics.generate_instance()
    solve_status, solve_time = logistics.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")