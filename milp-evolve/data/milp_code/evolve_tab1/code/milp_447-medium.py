import random
import time
import numpy as np
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
        leftover_nodes (-self.degrees).argsort().tolist()
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
    def erdos_renyi(number_of_nodes, edge_probability):
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

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

        transportation_capacity = np.random.randint(50, 200, size=n_locations)
        team_demand = np.random.randint(5, 20, size=n_teams)

        graph = Graph.barabasi_albert(n_locations, self.affinity)
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
            'adjacency': adjacency
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

        z = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(n_locations)}

        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations)) + \
                     quicksum(z[j] * activation_costs[j] for j in range(n_locations))

        model.setObjective(total_cost, "minimize")

        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) == 1, name=f"team_assignment_{i}")

        min_teams_per_location = 5
        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * team_demand[i] for i in range(n_teams)) <= transportation_capacity[j] * z[j],
                          name=f"transportation_capacity_{j}")

            model.addCons(quicksum(x[i, j] for i in range(n_teams)) >= min_teams_per_location * z[j],
                          name=f"min_teams_to_activate_{j}")

        lambda_vars = []
        cliques = self.efficient_greedy_clique_partition(adjacency)
        for count, clique in enumerate(cliques):
            lambda_vars.append(model.addVar(vtype="B", name=f"lambda_{count}"))
            for node in clique:
                model.addCons(z[node] <= lambda_vars[count], name=f"convex_{count}_{node}")

        for u in range(n_locations):
            for v in range(u + 1, n_locations):
                if adjacency[u, v] == 1:
                    model.addCons(z[u] + z[v] <= 1, name=f"adjacency_{u}_{v}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

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
        'min_teams': 60,
        'max_teams': 168,
        'min_locations': 8,
        'max_locations': 90,
        'affinity': 1,
    }

    deployment = ComplexTeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")