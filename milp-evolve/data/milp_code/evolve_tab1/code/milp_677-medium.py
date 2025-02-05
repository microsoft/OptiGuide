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
            G.nodes[node]['start_time'] = np.random.randint(0, 50)
            G.nodes[node]['end_time'] = G.nodes[node]['start_time'] + np.random.randint(10, 20)
        for u, v in G.edges:
            G[u][v]['handling_time'] = np.random.randint(1, 10)
            G[u][v]['capacity'] = np.random.randint(20, 50)
            G[u][v]['weight_limit'] = np.random.randint(100, 200)  # Simplified attributes
            G[u][v]['max_flow'] = np.random.randint(10, 30)
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
        time_vars = {}
        delay_penalty_vars = {}
        
        # Decision variables for knapsack allocation
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")
        
        # Decision variables for routing
        for i in range(number_of_items):
            for u, v in G.edges():
                flow_vars[(i, u, v)] = model.addVar(vtype="B", name=f"flow_{i}_{u}_{v}")
                handle_cost_vars[(i, u, v)] = model.addVar(vtype="B", name=f"handle_cost_{i}_{u}_{v}")
        
        # Time variables for arrival times
        for i in range(number_of_items):
            for u in G.nodes():
                time_vars[(i, u)] = model.addVar(vtype="C", name=f"time_{i}_{u}")

        # Penalty variables for delay
        for i in range(number_of_items):
            for u in G.nodes():
                delay_penalty_vars[(i, u)] = model.addVar(vtype="C", name=f"delay_{i}_{u}", lb=0)

        # Objective: Maximize total profit minus handling and delay costs
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        objective_expr -= quicksum(G[u][v]['handling_time'] * handle_cost_vars[(i, u, v)] for i in range(number_of_items) for u, v in G.edges())
        objective_expr -= quicksum(delay_penalty_vars[(i, u)] for i in range(number_of_items) for u in G.nodes())

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

        # Weight limit constraints
        for u, v in G.edges():
            total_weight = quicksum(weights[i] * flow_vars[(i, u, v)] for i in range(number_of_items))
            model.addCons(total_weight <= G[u][v]['weight_limit'], f"WeightLimit_{u}_{v}")

        # Time window constraints
        for i in range(number_of_items):
            for u in G.nodes():
                start = G.nodes[u]['start_time']
                end = G.nodes[u]['end_time']
                model.addCons(
                    time_vars[(i, u)] >= start,
                    f"TimeWindowStart_{i}_{u}"
                )
                model.addCons(
                    time_vars[(i, u)] <= end,
                    f"TimeWindowEnd_{i}_{u}"
                )

        # Delay penalty constraints
        for i in range(number_of_items):
            for u in G.nodes():
                model.addCons(
                    delay_penalty_vars[(i, u)] >= time_vars[(i, u)] - G.nodes[u]['end_time'],
                    f"DelayPenalty_{i}_{u}"
                )

        # Maximum flow constraints
        for u, v in G.edges():
            total_flow = quicksum(flow_vars[(i, u, v)] for i in range(number_of_items))
            model.addCons(total_flow <= G[u][v]['max_flow'], f"MaxFlow_{u}_{v}")

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