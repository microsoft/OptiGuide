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

    def efficient_greedy_clique_partition(self):
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

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

class LogisticsPackageRouting:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_nodes > 0 and self.n_customers > 0
        assert self.min_depot_cost >= 0 and self.max_depot_cost >= self.min_depot_cost
        assert self.min_customer_cost >= 0 and self.max_customer_cost >= self.min_customer_cost
        assert self.min_depot_capacity > 0 and self.max_depot_capacity >= self.min_depot_capacity

        depot_costs = np.random.randint(self.min_depot_cost, self.max_depot_cost + 1, self.n_nodes)
        customer_costs = np.random.randint(self.min_customer_cost, self.max_customer_cost + 1, (self.n_nodes, self.n_customers))
        capacities = np.random.randint(self.min_depot_capacity, self.max_depot_capacity + 1, self.n_nodes)
        demands = np.random.randint(1, 10, self.n_customers)

        graph = Graph.barabasi_albert(self.n_nodes, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_travel_times = np.random.randint(1, 10, size=len(graph.edges))
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(self.n_nodes):
            if node not in used_nodes:
                inequalities.add((node,))
        
        return {
            "depot_costs": depot_costs,
            "customer_costs": customer_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_travel_times": edge_travel_times,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        depot_costs = instance['depot_costs']
        customer_costs = instance['customer_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_travel_times = instance['edge_travel_times']
        
        model = Model("LogisticsPackageRouting")
        n_nodes = len(depot_costs)
        n_customers = len(customer_costs[0])
        
        # Decision variables
        depot_open = {c: model.addVar(vtype="B", name=f"DepotOpen_{c}") for c in range(n_nodes)}
        customer_served = {(c, r): model.addVar(vtype="B", name=f"Depot_{c}_Customer_{r}") for c in range(n_nodes) for r in range(n_customers)}
        route_setup = {edge: model.addVar(vtype="B", name=f"RouteSetup_{edge[0]}_{edge[1]}") for edge in graph.edges}
        
        # Objective: minimize the total cost including depot costs, customer servicing costs, and travel distances
        model.setObjective(
            quicksum(depot_costs[c] * depot_open[c] for c in range(n_nodes)) +
            quicksum(customer_costs[c, r] * customer_served[c, r] for c in range(n_nodes) for r in range(n_customers)) +
            quicksum(edge_travel_times[i] * route_setup[edge] for i, edge in enumerate(graph.edges)), "minimize"
        )
        
        # Constraints: Each customer is served by at least one depot
        for r in range(n_customers):
            model.addCons(quicksum(customer_served[c, r] for c in range(n_nodes)) >= 1, f"Customer_{r}_Service")
        
        # Constraints: Only open depots can serve customers
        for c in range(n_nodes):
            for r in range(n_customers):
                model.addCons(customer_served[c, r] <= depot_open[c], f"Depot_{c}_Serve_{r}")
        
        # Constraints: Depots cannot exceed their capacity
        for c in range(n_nodes):
            model.addCons(quicksum(demands[r] * customer_served[c, r] for r in range(n_customers)) <= capacities[c], f"Depot_{c}_Capacity")
        
        # Constraints: Depot Route Cliques
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(depot_open[node] for node in group) <= 1, f"Clique_{count}")
        
        # Adding complexity: additional clique constraints
        complexities = list(inequalities)  # convert set to list to manipulate
        
        # Additional larger cliques formed from subsets of existing cliques
        # Let's assume the complexity involves adding cliques of size 3 if possible
        new_cliques = []
        for clique in complexities:
            if isinstance(clique, tuple) and len(clique) >= 3:
                new_cliques.extend(list(combinations(clique, 3)))
        
        complexity_vars = {idx: model.addVar(vtype="B", name=f"Complexity_{idx}") for idx in range(len(new_cliques))}
        
        for idx, clique in enumerate(new_cliques):
            model.addCons(quicksum(depot_open[node] for node in clique) <= complexity_vars[idx], f"Complexity_Clique_{idx}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 37,
        'n_customers': 112,
        'min_customer_cost': 686,
        'max_customer_cost': 3000,
        'min_depot_cost': 1686,
        'max_depot_cost': 5000,
        'min_depot_capacity': 277,
        'max_depot_capacity': 2520,
        'affinity': 5,
    }

    logistics_optimizer = LogisticsPackageRouting(parameters, seed)
    instance = logistics_optimizer.generate_instance()
    solve_status, solve_time, objective_value = logistics_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")