import random
import time
import numpy as np
from pyscipopt import Model, quicksum
from itertools import combinations

############# Helper function #############
class Graph:
    """
    Helper function: Container for a graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def efficient_greedy_clique_partition(self):
        """
        Partition the graph into cliques using an efficient greedy algorithm.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.
        """
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

############# Helper function #############

class MultipleKnapsackWithIndependentSet:
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

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (self.max_range-self.min_range), 1),
                               weights + (self.max_range-self.min_range)]))

        elif self.scheme == 'strongly correlated':
            profits = weights + (self.max_range - self.min_range) / 10

        elif self.scheme == 'subset-sum':
            profits = weights

        else:
            raise NotImplementedError

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()
        
        # Generate random environmental impacts for each item
        environmental_impacts = np.random.randint(self.env_min, self.env_max, self.number_of_items)

        # Generate a random graph for independent set constraint
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.number_of_items, self.edge_probability)
        else:
            raise ValueError("Unsupported graph type.")

        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        # Put trivial inequalities for nodes that didn't appear
        # in the constraints, otherwise SCIP will complain
        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(min(self.number_of_items, 10)):  # to ensure all nodes are accounted for
            if node not in used_nodes:
                inequalities.add((node,))

        res = {'weights': weights, 
               'profits': profits, 
               'capacities': capacities,
               'environmental_impacts': environmental_impacts,
               'graph': graph,
               'inequalities': inequalities}

        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        environmental_impacts = instance['environmental_impacts']
        inequalities = instance['inequalities']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)
        
        model = Model("MultipleKnapsackWithIndependentSet")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Objective: Maximize total profit and minimize environmental impact
        profit_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))
        env_impact_expr = quicksum(environmental_impacts[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks))

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
        
        # Environmental Impact Constraint: Total environmental impact must be below a certain threshold
        model.addCons(
            env_impact_expr <= self.max_env_impact,
            "MaxEnvironmentalImpact"
        )

        # Independent Set Constraints: Ensure selected items form an independent set in the given graph
        for count, group in enumerate(inequalities):
            for j in range(number_of_knapsacks):
                model.addCons(
                    quicksum(var_names[(node, j)] for node in group) <= 1,
                    f"IndependentSet_{count}_{j}"
                )

        # Dual Objective: Maximize profit and minimize environmental impact
        alpha = 0.5  # Trade-off parameter, adjust alpha to balance between profit maximization and environmental impact minimization
        model.setObjective((1 - alpha) * profit_expr - alpha * env_impact_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 75,
        'number_of_knapsacks': 7,
        'min_range': 3,
        'max_range': 75,
        'scheme': 'weakly correlated',
        'env_min': 0,
        'env_max': 100,
        'max_env_impact': 1125,
        'graph_type': 'erdos_renyi',
        'edge_probability': 0.73,
    }

    knapsack_with_independent_set = MultipleKnapsackWithIndependentSet(parameters, seed=seed)
    instance = knapsack_with_independent_set.generate_instance()
    solve_status, solve_time = knapsack_with_independent_set.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")