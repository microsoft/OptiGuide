# Import necessary libraries
import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

# Define the Graph class
class Graph:
    """
    Helper function: Container for a graph.
    """
    def __init__(self, number_of_nodes, edges):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.
        """
        edges = set()
        for edge in combinations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
        return Graph(number_of_nodes, edges)

# Define the Main MILP class for Complex Hub Location with Auction and Dependencies
class ComplexHubLocationWithAuctionAndDependencies:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        return Graph.erdos_renyi(self.n_nodes, self.edge_probability)

    def generate_instance(self):
        graph = self.generate_graph()
        opening_costs = np.random.randint(20, 70, size=graph.number_of_nodes)
        connection_costs = np.random.randint(1, 15, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Simulate auction bid generation
        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_nodes)
        bids = []
        for _ in range(self.n_bids):
            private_interests = np.random.rand(self.n_nodes)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)
            initial_node = np.random.choice(self.n_nodes, p=private_interests / private_interests.sum())
            bundle_mask = np.zeros(self.n_nodes, dtype=bool)
            bundle_mask[initial_node] = True
            while np.random.rand() < self.add_item_prob and bundle_mask.sum() < self.n_nodes:
                next_node = np.random.choice(self.n_nodes, p=(private_interests * ~bundle_mask) / (private_interests * ~bundle_mask).sum())
                bundle_mask[next_node] = True
            bundle = np.nonzero(bundle_mask)[0]
            revenue = private_values[bundle].sum() + len(bundle) ** (1 + self.additivity)
            if revenue > 0:
                bids.append((list(bundle), revenue))
        bids_per_node = [[] for _ in range(self.n_nodes)]
        for i, bid in enumerate(bids):
            bundle, revenue = bid
            for node in bundle:
                bids_per_node[node].append(i)

        # Generate dependency matrix
        dependencies = np.random.binomial(1, self.dependency_prob, size=(self.n_nodes, self.n_nodes))

        res = {
            'graph': graph,
            'opening_costs': opening_costs,
            'connection_costs': connection_costs,
            'bids': bids,
            'bids_per_node': bids_per_node,
            'dependencies': dependencies
        }
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        opening_costs = instance['opening_costs']
        connection_costs = instance['connection_costs']
        bids = instance['bids']
        bids_per_node = instance['bids_per_node']
        dependencies = instance['dependencies']

        model = Model("ComplexHubLocationWithAuctionAndDependencies")
        
        # Add variables for hubs and routing
        hub_vars = {node: model.addVar(vtype="B", name=f"hub_{node}") for node in graph.nodes}
        routing_vars = {(i, j): model.addVar(vtype="B", name=f"route_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        bid_vars = {i: model.addVar(vtype="B", name=f"bid_{i}") for i in range(len(bids))}
        penalty_vars = {i: model.addVar(vtype="B", name=f"penalty_{i}") for i in graph.nodes}

        # Connection Constraints
        for node in graph.nodes:
            model.addCons(quicksum(routing_vars[node, hub] for hub in graph.nodes) == 1, name=f"ConnectionConstraints_{node}")

        # Ensure that routing is to an opened hub
        for node in graph.nodes:
            for hub in graph.nodes:
                model.addCons(routing_vars[node, hub] <= hub_vars[hub], name=f"ServiceProvision_{node}_{hub}")

        # Ensure that items are in at most one bundle
        for node, bid_indices in enumerate(bids_per_node):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= 1, f"NodeBundling_{node}")

        # Dependency Constraints
        for i in graph.nodes:
            for j in graph.nodes:
                if dependencies[i, j] == 1:
                    model.addCons(hub_vars[i] <= hub_vars[j] + penalty_vars[i], f"Dependency_{i}_{j}")

        # Objective function: Minimize the total cost while maximizing revenue and minimizing penalties
        hub_opening_cost = quicksum(hub_vars[node] * opening_costs[node] for node in graph.nodes)
        connection_total_cost = quicksum(routing_vars[i, j] * connection_costs[i, j] for i in graph.nodes for j in graph.nodes)
        total_revenue = quicksum(revenue * bid_vars[i] for i, (bundle, revenue) in enumerate(bids))
        penalty_cost = quicksum(penalty_vars[i] * self.penalty_cost for i in graph.nodes)
        total_cost = hub_opening_cost + connection_total_cost - total_revenue + penalty_cost

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 100,
        'edge_probability': 0.38,
        'n_bids': 3000,
        'min_value': 2,
        'max_value': 105,
        'value_deviation': 0.45,
        'additivity': 0.1,
        'add_item_prob': 0.31,
        'dependency_prob': 0.66,
        'penalty_cost': 10,
    }

    hub_location_problem = ComplexHubLocationWithAuctionAndDependencies(parameters, seed=seed)
    instance = hub_location_problem.generate_instance()
    solve_status, solve_time = hub_location_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")