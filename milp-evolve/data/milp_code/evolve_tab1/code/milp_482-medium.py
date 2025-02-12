import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

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
    def random_graph(number_of_nodes, neighbor_probability):
        G = nx.watts_strogatz_graph(number_of_nodes, k=4, p=neighbor_probability)
        edges = set(G.edges)
        degrees = np.array([val for (node, val) in G.degree()])
        neighbors = {node: set(G.neighbors(node)) for node in range(number_of_nodes)}
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class ComplexTeamDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        n_teams = random.randint(self.min_teams, self.max_teams)
        n_locations = random.randint(self.min_locations, self.max_locations)

        team_costs = np.random.randint(10, 100, size=(n_teams, n_locations))
        activation_costs = np.random.randint(20, 200, size=n_locations)

        transportation_capacity = np.random.randint(50, 300, size=n_locations)  # Increased upper boundary for complexity
        team_demand = np.random.randint(5, 20, size=n_teams)

        graph = Graph.random_graph(n_locations, self.neighbor_probability)
        adjacency = np.zeros((n_locations, n_locations), dtype=int)
        for edge in graph.edges:
            adjacency[edge[0], edge[1]] = 1
            adjacency[edge[1], edge[0]] = 1

        res = {
            'n_teams': n_teams,
            'n_locations': n_locations,
            'team_costs': team_costs,
            'activation_costs': activation_costs,
            'transportation_capacity': transportation_capacity,
            'team_demand': team_demand,
            'adjacency': adjacency,
        }
        return res

    def solve(self, instance):
        n_teams = instance['n_teams']
        n_locations = instance['n_locations']
        team_costs = instance['team_costs']
        activation_costs = instance['activation_costs']
        transportation_capacity = instance['transportation_capacity']
        team_demand = instance['team_demand']
        adjacency = instance['adjacency']

        model = Model("ComplexTeamDeployment")

        x = {}
        for i in range(n_teams):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        capacity_usage = {j: model.addVar(vtype="I", lb=0, name=f"capacity_usage_{j}") for j in range(n_locations)}

        # Multi-objective: Minimize costs and minimize maximum demand-capacity ratio
        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations))
        total_cost += quicksum(capacity_usage[j] * activation_costs[j] for j in range(n_locations))

        # Additional objective: Minimize the maximum demand-capacity ratio across all locations
        max_demand_capacity_ratio = model.addVar(vtype="C", name="max_demand_capacity_ratio")
        model.setObjective(total_cost + max_demand_capacity_ratio, "minimize")

        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * team_demand[i] for i in range(n_teams)) <= transportation_capacity[j],
                          name=f"capacity_{j}")

            # Introduce inter-location constraints based on adjacency (correlation costs)
            for k in range(n_locations):
                if adjacency[j, k] == 1:
                    model.addCons(capacity_usage[j] >= 0.5 * capacity_usage[k], name=f"adjacency_relax_{j}_{k}")

        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) == 1, name=f"team_assignment_{i}")

        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] for i in range(n_teams)) >= 1, name=f"non_empty_location_{j}")

            # Additional constraints for maximizing overall team coverage
            model.addCons(capacity_usage[j] == quicksum(x[i, j] * team_demand[i] for i in range(n_teams)), 
                          name=f"use_capacity_correctly_{j}")
            model.addCons(capacity_usage[j] <= max_demand_capacity_ratio * transportation_capacity[j], name=f"max_ratio_limit_{j}")

        # Add clique constraints 
        cliques = self.efficient_greedy_clique_partition(adjacency)
        lambda_vars = [model.addVar(vtype="B", name=f"lambda_{c}") for c in range(len(cliques))]
        for c, clique in enumerate(cliques):
            for node in clique:
                model.addCons(capacity_usage[node] <= lambda_vars[c] * transportation_capacity[node], name=f"clique_{c}_{node}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        solve_status = model.getStatus()
        solve_time = end_time - start_time

        return solve_status, solve_time

    def efficient_greedy_clique_partition(self, adjacency):
        n_locations = adjacency.shape[0]
        degrees = np.sum(adjacency, axis=1)
        cliques = []
        leftover_nodes = (-degrees).argsort().tolist()
        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = set(np.where(adjacency[clique_center] == 1)[0]).intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -degrees[x])
            for neighbor in densest_neighbors:
                if all(adjacency[neighbor, clique_node] == 1 for clique_node in clique):
                    clique.add(neighbor)
            cliques.append(list(clique))
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_teams': 15,
        'max_teams': 42,
        'min_locations': 0,
        'max_locations': 45,
        'neighbor_probability': 0.31,
    }

    deployment = ComplexTeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")