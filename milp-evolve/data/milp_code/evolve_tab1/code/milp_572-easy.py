import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MultipleKnapsackWithNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
        
    def generate_instance(self):
        weights = np.random.randint(self.min_range, self.max_range, self.number_of_items)
        profits = np.apply_along_axis(lambda x: np.random.randint(x[0], x[1]), axis=0,
                                      arr=np.vstack([np.maximum(weights - (self.max_range-self.min_range), 1), weights + (self.max_range-self.min_range)]))
        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        G = nx.erdos_renyi_graph(n=self.network_nodes, p=self.network_prob, directed=True, seed=self.seed)
        for node in G.nodes:
            G.nodes[node]['cargo_weight'] = np.random.randint(10, 100)
        for u, v in G.edges:
            G[u][v]['handling_time'] = np.random.randint(1, 10)
            G[u][v]['capacity'] = np.random.randint(20, 50)
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)

        return {'weights': weights, 
                'profits': profits, 
                'capacities': capacities, 
                'graph': G, 
                'invalid_edges': E_invalid}

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        G = instance['graph']
        E_invalid = instance['invalid_edges']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        number_of_nodes = len(G.nodes)

        model = Model("MultipleKnapsackWithNetwork")
        var_names = {}
        flow_vars = {}
        handle_cost_vars = {}
        
        # Decision variables for knapsack allocation
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
        
        # Decision variables for routing
        for i in range(number_of_items):
            for u, v in G.edges():
                flow_vars[(i, u, v)] = model.addVar(vtype="B", name=f"flow_{i}_{u}_{v}")
                handle_cost_vars[(i, u, v)] = model.addVar(vtype="B", name=f"handle_cost_{i}_{u}_{v}")

        # Objective: Maximize total profit minus handling cost
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        objective_expr -= quicksum(G[u][v]['handling_time'] * handle_cost_vars[(i, u, v)] for i in range(number_of_items) for u, v in G.edges())

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= 1,
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )

        # Routing constraints: ensure valid routing paths
        for i in range(number_of_items):
            for u, v in G.edges():
                if (u, v) in E_invalid:
                    model.addCons(flow_vars[(i, u, v)] == 0, f"InvalidEdge_{i}_{u}_{v}")
                else:
                    model.addCons(flow_vars[(i, u, v)] <= 1, f"ValidEdge_{i}_{u}_{v}")

        # Ensure nodes match routing requirements
        for i in range(number_of_items):
            for u in G.nodes():
                in_flow = quicksum(flow_vars[(i, w, u)] for w in G.predecessors(u))
                out_flow = quicksum(flow_vars[(i, u, w)] for w in G.successors(u))
                node_load = quicksum(handle_cost_vars[(i, u, v)] for v in G.successors(u))
                model.addCons(
                    in_flow - out_flow - node_load == 0,
                    name=f"FlowConservation_{i}_{u}"
                )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 200,
        'number_of_knapsacks': 10,
        'min_range': 10,
        'max_range': 30,
        'scheme': 'weakly correlated',
        'network_nodes': 15,
        'network_prob': 0.15,
        'exclusion_rate': 0.1,
    }

    knapsack_network = MultipleKnapsackWithNetwork(parameters, seed=seed)
    instance = knapsack_network.generate_instance()
    solve_status, solve_time = knapsack_network.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")