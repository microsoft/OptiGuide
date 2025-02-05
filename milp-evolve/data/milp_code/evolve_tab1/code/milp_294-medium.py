import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
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
############# Helper function #############

class IndependentSet:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        graph = self.generate_graph()
        node_weights = np.random.randint(1, self.max_weight, self.n_nodes)

        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        res = {'graph': graph,
               'inequalities': inequalities,
               'node_weights': node_weights}

        # Generating new data for GDO integration:
        market_vars = {(u, v): np.random.randint(1, 20) for u, v in graph.edges}
        zone_vars = {(u, v): np.random.randint(1, 20) for u, v in graph.edges}
        congestion_vars = {(u, v): np.random.randint(0, 2) for u, v in graph.edges}
        nutrient_demand_vars = {node: np.random.randint(1, 100) for node in graph.nodes}
        transport_costs = {(u, v): np.random.randint(1, 20) for u, v in graph.edges}

        res.update({
            'market_vars': market_vars,
            'zone_vars': zone_vars,
            'congestion_vars': congestion_vars,
            'nutrient_demand_vars': nutrient_demand_vars,
            'transport_costs': transport_costs,
        })
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        node_weights = instance['node_weights']

        market_vars = instance['market_vars']
        zone_vars = instance['zone_vars']
        congestion_vars = instance['congestion_vars']
        nutrient_demand_vars = instance['nutrient_demand_vars']
        transport_costs = instance['transport_costs']

        model = Model("IndependentSet")
        var_names = {}

        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        for count, group in enumerate(inequalities):
            if len(group) > 1:
                model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        # New variables for GDO
        market_vars_pyscipopt = {f"m{u}_{v}": model.addVar(vtype="B", name=f"m{u}_{v}") for u, v in graph.edges}
        zone_vars_pyscipopt = {f"z{u}_{v}": model.addVar(vtype="B", name=f"z{u}_{v}") for u, v in graph.edges}
        congestion_vars_pyscipopt = {f"cong_{u}_{v}": model.addVar(vtype="B", name=f"cong_{u}_{v}") for u, v in graph.edges}
        cost_vars_pyscipopt = {f"c{node}": model.addVar(vtype="C", name=f"c{node}") for node in graph.nodes}
        nutrient_demand_vars_pyscipopt = {f"d{node}": model.addVar(vtype="C", name=f"d{node}") for node in graph.nodes}
        weekly_budget_pyscipopt = model.addVar(vtype="C", name="weekly_budget")

        # New constraints and objective function for integrated model
        combined_objective_expr = quicksum(
            node_weights[node] * var_names[node] for node in graph.nodes
        ) + quicksum(
            nutrient_demand_vars[node] * var_names[node] for node in graph.nodes
        ) - quicksum(
            transport_costs[(u, v)] * market_vars_pyscipopt[f"m{u}_{v}"] for u, v in graph.edges
        ) - quicksum(
            market_vars[(u, v)] * congestion_vars_pyscipopt[f"cong_{u}_{v}"] for u, v in graph.edges
        ) - quicksum(
            cost_vars_pyscipopt[f"c{node}"] for node in graph.nodes
        )

        model.setObjective(combined_objective_expr, "maximize")

        # Constraints for GDO integration
        for u, v in graph.edges:
            model.addCons(var_names[u] + var_names[v] - market_vars_pyscipopt[f"m{u}_{v}"] <= 1)
            model.addCons(var_names[u] + var_names[v] <= 1 + zone_vars_pyscipopt[f"z{u}_{v}"])
            model.addCons(var_names[u] + var_names[v] >= 2 * zone_vars_pyscipopt[f"z{u}_{v}"])
            model.addCons(market_vars_pyscipopt[f"m{u}_{v}"] >= congestion_vars_pyscipopt[f"cong_{u}_{v}"])

        for node in graph.nodes:
            model.addCons(cost_vars_pyscipopt[f"c{node}"] + nutrient_demand_vars_pyscipopt[f"d{node}"] == 0)

        model.addCons(weekly_budget_pyscipopt <= self.weekly_budget_limit)

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 315,
        'edge_probability': 0.17,
        'affinity': 81,
        'graph_type': 'barabasi_albert',
        'max_weight': 1366,
        'weekly_budget_limit': 2000,
    }

    independent_set = IndependentSet(parameters, seed=seed)
    instance = independent_set.generate_instance()
    solve_status, solve_time = independent_set.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")