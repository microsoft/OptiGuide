import random
import time
import numpy as np
import networkx as nx
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
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """
        Generate an Erdős-Rényi random graph with a given edge probability.
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
        Generate a Barabási-Albert random graph with a given affinity.
        """
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

class NetworkMaintenance:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
            
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_task_assignment_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_tasks, 1) - rand(1, self.n_servers))**2 +
            (rand(self.n_tasks, 1) - rand(1, self.n_servers))**2
        )
        return costs

    def generate_instance(self):
        task_demands = self.randint(self.n_tasks, self.task_demand_interval)
        server_capacities = self.randint(self.n_servers, self.server_capacity_interval)
        maintenance_costs = (
            self.randint(self.n_servers, self.maintenance_cost_scale_interval) * np.sqrt(server_capacities) +
            self.randint(self.n_servers, self.maintenance_cost_cste_interval)
        )
        task_assignment_costs = self.unit_task_assignment_costs() * task_demands[:, np.newaxis]

        server_capacities = server_capacities * self.ratio * np.sum(task_demands) / np.sum(server_capacities)
        server_capacities = np.round(server_capacities)
        
        res = {
            'task_demands': task_demands,
            'server_capacities': server_capacities,
            'maintenance_costs': maintenance_costs,
            'task_assignment_costs': task_assignment_costs
        }
        
        n_edges = (self.n_servers * (self.n_servers - 1)) // 4
        G = nx.erdos_renyi_graph(self.n_servers, p=n_edges/(self.n_servers*(self.n_servers-1)))
        clusters = list(nx.find_cliques(G))
        random_clusters = [cl for cl in clusters if len(cl) > 2][:self.max_clusters]
        res['clusters'] = random_clusters
        
        graph = Graph.erdos_renyi(self.n_servers, self.edge_probability)
        
        node_existence_prob = np.random.uniform(0.8, 1, self.n_servers)
        node_weights = np.random.randint(1, self.max_weight, self.n_servers)
        knapsack_capacity = np.random.randint(self.min_capacity, self.max_capacity)
        
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)
                
        res.update({
            'independent_set_graph': graph,
            'inequalities': inequalities,
            'node_existence_prob': node_existence_prob,
            'node_weights': node_weights,
            'knapsack_capacity': knapsack_capacity
        })
        
        return res

    def solve(self, instance):
        task_demands = instance['task_demands']
        server_capacities = instance['server_capacities']
        maintenance_costs = instance['maintenance_costs']
        task_assignment_costs = instance['task_assignment_costs']
        clusters = instance['clusters']
        
        inequalities = instance['inequalities']
        node_existence_prob = instance['node_existence_prob']
        node_weights = instance['node_weights']
        knapsack_capacity = instance['knapsack_capacity']
        
        n_tasks = len(task_demands)
        n_servers = len(server_capacities)
        M = np.max(task_demands)
        
        model = Model("NetworkMaintenance")
        
        # Decision variables
        operational_servers = {j: model.addVar(vtype="B", name=f"Operational_{j}") for j in range(n_servers)}
        assign = {(i, j): model.addVar(vtype="C", name=f"Assign_{i}_{j}") for i in range(n_tasks) for j in range(n_servers)}
        independent_set_vars = {j: model.addVar(vtype="B", name=f"IndependentSet_{j}") for j in range(n_servers)}
        
        # Objective: minimize the total cost including maintenance, assignment, and maximize node existence
        objective_expr = quicksum(maintenance_costs[j] * operational_servers[j] for j in range(n_servers)) + \
                         quicksum(task_assignment_costs[i, j] * assign[i, j] for i in range(n_tasks) for j in range(n_servers)) - \
                         quicksum(node_existence_prob[j] * independent_set_vars[j] for j in range(n_servers))
        
        # Constraints: task demand must be met
        for i in range(n_tasks):
            model.addCons(quicksum(assign[i, j] for j in range(n_servers)) >= 1, f"TaskDemand_{i}")
        
        # Constraints: server capacity limits
        for j in range(n_servers):
            model.addCons(quicksum(assign[i, j] * task_demands[i] for i in range(n_tasks)) <= server_capacities[j] * operational_servers[j], f"ServerCapacity_{j}")
        
        # Constraints: tightening constraints
        total_task_demand = np.sum(task_demands)
        model.addCons(quicksum(server_capacities[j] * operational_servers[j] for j in range(n_servers)) >= total_task_demand, "TotalTaskDemand")
        
        for i in range(n_tasks):
            for j in range(n_servers):
                model.addCons(assign[i, j] <= operational_servers[j] * M, f"Tightening_{i}_{j}")

        for idx, cluster in enumerate(clusters[:self.max_clusters]):
            model.addCons(quicksum(operational_servers[j] for j in cluster) <= self.compliance_limit, f"Cluster_{idx}")
        
        # Independent set constraints
        for count, group in enumerate(inequalities):
            if len(group) > 1:
                model.addCons(quicksum(independent_set_vars[node] for node in group) <= 1, name=f"IndependentClique_{count}")

        model.addCons(quicksum(node_weights[j] * independent_set_vars[j] for j in range(n_servers)) <= knapsack_capacity, name="Knapsack")
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_tasks': 75,
        'n_servers': 75,
        'task_demand_interval': (2, 20),
        'server_capacity_interval': (35, 563),
        'maintenance_cost_scale_interval': (400, 444),
        'maintenance_cost_cste_interval': (0, 11),
        'ratio': 18.8,
        'continuous_assignment': 27,
        'max_clusters': 270,
        'compliance_limit': 9,
        'edge_probability': 0.52,
        'max_weight': 506,
        'min_capacity': 10000,
        'max_capacity': 15000,
    }

    network_maintenance = NetworkMaintenance(parameters, seed=seed)
    instance = network_maintenance.generate_instance()
    solve_status, solve_time = network_maintenance.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")