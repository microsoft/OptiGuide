import random
import time
import numpy as np
import scipy
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum

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

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        """
        Generate a Barabási-Albert random graph with a given edge probability.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
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

class IndependentSetWithSetCover:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
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
        for node in range(10):
            if node not in used_nodes:
                inequalities.add((node,))

        # Generate set cover data
        nnzrs = int(self.n_required_nodes * self.n_sets * self.density)
        indices = np.random.choice(self.n_sets, size=nnzrs)  # random column indexes
        indices[:2 * self.n_sets] = np.repeat(np.arange(self.n_sets), 2)  # force at least 2 rows per col
        _, col_nrows = np.unique(indices, return_counts=True)
        indices[:self.n_required_nodes] = np.random.permutation(self.n_required_nodes)  # force at least 1 column per row
        i = 0
        indptr = [0]
        for n in col_nrows:
            # empty column, fill with random rows
            if i >= self.n_required_nodes:
                indices[i:i+n] = np.random.choice(self.n_required_nodes, size=n, replace=False)

            # partially filled column, complete with random rows among remaining ones
            elif i + n > self.n_required_nodes:
                remaining_rows = np.setdiff1d(np.arange(self.n_required_nodes), indices[i:self.n_required_nodes], assume_unique=True)
                indices[self.n_required_nodes:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_required_nodes, replace=False)

            i += n
            indptr.append(i)

        # objective coefficients for sets
        c = np.random.randint(self.max_coef, size=self.n_sets) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_required_nodes, self.n_sets)
        ).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        res = {
            'graph': graph,
            'inequalities': inequalities,
            'set_cover_data': {
                'c': c,
                'indptr_csr': indptr_csr,
                'indices_csr': indices_csr
            }
        }

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']        
        set_cover_data = instance['set_cover_data']
        c = set_cover_data['c']
        indptr_csr = set_cover_data['indptr_csr']
        indices_csr = set_cover_data['indices_csr']

        model = Model("IndependentSetWithSetCover")
        var_names = {}
        set_vars = {}

        # Create variables for independent set
        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        # Create constraints for independent set
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        # Create variables for set cover
        for j in range(self.n_sets):
            set_vars[j] = model.addVar(vtype="B", name=f"y_{j}", obj=c[j])

        # Add constraints to ensure each required node is covered
        for row in range(self.n_required_nodes):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(set_vars[j] for j in cols) >= 1, name=f"cover_{row}")

        # Update objective to combine both Independent Set and Set Cover aspects
        objective_expr = quicksum(var_names[node] for node in graph.nodes) - quicksum(set_vars[j] * c[j] for j in range(self.n_sets))
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 750,
        'edge_probability': 0.25,
        'affinity': 4, 
        'graph_type': 'barabasi_albert',
        'n_required_nodes': 350,
        'n_sets': 400,
        'density': 0.05,
        'max_coef': 100,
    }

    independent_set_with_set_cover_problem = IndependentSetWithSetCover(parameters, seed=seed)
    instance = independent_set_with_set_cover_problem.generate_instance()
    solve_status, solve_time = independent_set_with_set_cover_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")