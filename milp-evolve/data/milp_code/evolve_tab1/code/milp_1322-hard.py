import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class FarmersMarketPlacement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.num_markets > 0 and self.num_customers > 0
        assert self.min_management_cost >= 0 and self.max_management_cost >= self.min_management_cost
        assert self.min_customer_cost >= 0 and self.max_customer_cost >= self.min_customer_cost
        assert self.min_market_coverage > 0 and self.max_market_coverage >= self.min_market_coverage

        management_costs = np.random.randint(self.min_management_cost, self.max_management_cost + 1, self.num_markets)
        customer_costs = np.random.randint(self.min_customer_cost, self.max_customer_cost + 1, (self.num_markets, self.num_customers))
        coverages = np.random.randint(self.min_market_coverage, self.max_market_coverage + 1, self.num_markets)
        demands = np.random.randint(1, 10, self.num_customers)

        graph = Graph.barabasi_albert(self.num_markets, self.affinity)
        cliques = []
        for clique in nx.find_cliques(nx.Graph(graph.edges)):
            if len(clique) > 1:
                cliques.append(tuple(sorted(clique)))

        return {
            "management_costs": management_costs,
            "customer_costs": customer_costs,
            "coverages": coverages,
            "demands": demands,
            "cliques": cliques,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        management_costs = instance['management_costs']
        customer_costs = instance['customer_costs']
        coverages = instance['coverages']
        demands = instance['demands']
        cliques = instance['cliques']
        
        model = Model("FarmersMarketPlacement")
        num_markets = len(management_costs)
        num_customers = len(customer_costs[0])
        
        # Decision variables
        market_open = {m: model.addVar(vtype="B", name=f"MarketOpen_{m}") for m in range(num_markets)}
        customer_covered = {(m, r): model.addVar(vtype="B", name=f"Market_{m}_Customer_{r}") for m in range(num_markets) for r in range(num_customers)}
        
        # Objective: minimize the total management cost including market rents and transportation costs
        model.setObjective(
            quicksum(management_costs[m] * market_open[m] for m in range(num_markets)) +
            quicksum(customer_costs[m, r] * customer_covered[m, r] for m in range(num_markets) for r in range(num_customers)), "minimize"
        )
        
        # Constraints: Each customer is covered by at least one market stand
        for r in range(num_customers):
            model.addCons(quicksum(customer_covered[m, r] for m in range(num_markets)) >= 1, f"Customer_{r}_Coverage")
        
        # Constraints: Only open market stands can cover customers
        for m in range(num_markets):
            for r in range(num_customers):
                model.addCons(customer_covered[m, r] <= market_open[m], f"Market_{m}_Cover_{r}")
        
        # Constraints: Market stands cannot exceed their coverage limit
        for m in range(num_markets):
            model.addCons(quicksum(demands[r] * customer_covered[m, r] for r in range(num_customers)) <= coverages[m], f"Market_{m}_CoverageLimit")
        
        # Constraints: Market Clique Limits
        for count, clique in enumerate(cliques):
            model.addCons(quicksum(market_open[node] for node in clique) <= 1, f"CliqueRestriction_{count}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_markets': 30,
        'num_customers': 200,
        'min_customer_cost': 1000,
        'max_customer_cost': 5000,
        'min_management_cost': 500,
        'max_management_cost': 3000,
        'min_market_coverage': 50,
        'max_market_coverage': 200,
        'affinity': 5,
    }

    market_optimizer = FarmersMarketPlacement(parameters, seed)
    instance = market_optimizer.generate_instance()
    solve_status, solve_time, objective_value = market_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")