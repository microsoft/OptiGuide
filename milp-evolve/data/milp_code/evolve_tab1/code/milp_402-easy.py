import random
import time
import numpy as np
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

class OptimalWarehousingAndDistribution:
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

        product_service_prob = np.random.uniform(0.8, 1, self.n_nodes)
        storage_capacities = np.random.randint(1, self.max_capacity, self.n_nodes)
        storage_deviations = np.random.rand(self.n_nodes) * self.capacity_deviation_factor
        distribution_volume = np.random.randint(self.min_volume, self.max_volume)
        
        neighborhoods = graph.efficient_greedy_clique_partition()
        holistic_constraints = set(graph.edges)
        for neighborhood in neighborhoods:
            neighborhood = tuple(sorted(neighborhood))
            for edge in combinations(neighborhood, 2):
                holistic_constraints.remove(edge)
            if len(neighborhood) > 1:
                holistic_constraints.add(neighborhood)
        
        warehouses = self.warehouses
        distribution_points = self.distribution_points
        routes = [(i, j) for i in range(warehouses) for j in range(distribution_points)]
        warehouse_efficiency = np.random.randint(50, 150, size=warehouses)
        client_product_demands = np.random.randint(30, 100, size=distribution_points)
        transportation_cost = np.random.randint(1, 20, size=(warehouses, distribution_points))

        res = {'graph': graph,
               'holistic_constraints': holistic_constraints,
               'product_service_prob': product_service_prob,
               'storage_capacities': storage_capacities,
               'storage_deviations': storage_deviations,
               'distribution_volume': distribution_volume,
               'warehouse_efficiency': warehouse_efficiency,
               'client_product_demands': client_product_demands,
               'transportation_cost': transportation_cost,
               'routes': routes}

        min_transport_threshold = np.random.randint(5, 15, size=(warehouses, distribution_points))
        res['min_transport_threshold'] = min_transport_threshold

        distribution_capacities = np.random.randint(10, 50, size=warehouses)
        res['distribution_capacities'] = distribution_capacities
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        holistic_constraints = instance['holistic_constraints']
        product_service_prob = instance['product_service_prob']
        storage_capacities = instance['storage_capacities']
        storage_deviations = instance['storage_deviations']
        distribution_volume = instance['distribution_volume']

        warehouse_efficiency = instance['warehouse_efficiency']
        client_product_demands = instance['client_product_demands']
        transportation_cost = instance['transportation_cost']
        routes = instance['routes']
        warehouses = self.warehouses
        distribution_points = self.distribution_points
        min_transport_threshold = instance['min_transport_threshold']
        distribution_capacities = instance['distribution_capacities']

        model = Model("OptimalWarehousingAndDistribution")
        warehouse_assign_vars = {}
        transport_flow_vars = {}
        flow_active_vars = {}

        for node in graph.nodes:
            warehouse_assign_vars[node] = model.addVar(vtype="B", name=f"warehouse_assign_{node}")

        for count, group in enumerate(holistic_constraints):
            if len(group) > 1:
                model.addCons(quicksum(warehouse_assign_vars[node] for node in group) <= 1, name=f"HolisticConstraint_{count}")

        capacity_constraints = quicksum((storage_capacities[node] + storage_deviations[node]) * warehouse_assign_vars[node] for node in graph.nodes)
        model.addCons(capacity_constraints <= distribution_volume, name="CapacityConstraint")

        objective_expr = quicksum(product_service_prob[node] * warehouse_assign_vars[node] for node in graph.nodes)

        for i, j in routes:
            transport_flow_vars[(i, j)] = model.addVar(vtype="C", name=f"transport_flow_{i}_{j}")
            flow_active_vars[(i, j)] = model.addVar(vtype="B", name=f"flow_active_{i}_{j}")

        for i in range(warehouses):
            model.addCons(quicksum(transport_flow_vars[(i, j)] for j in range(distribution_points)) <= warehouse_efficiency[i], f"WarehouseEfficiency_{i}")

        for j in range(distribution_points):
            model.addCons(quicksum(transport_flow_vars[(i, j)] for i in range(warehouses)) >= client_product_demands[j], f"ClientProductDemand_{j}")

        for i, j in routes:
            model.addCons(transport_flow_vars[(i, j)] >= min_transport_threshold[i, j] * flow_active_vars[(i, j)], name=f"TransportFlow_{i}_{j}")

        cost_expr = quicksum(transport_flow_vars[(i, j)] * transportation_cost[i, j] for (i, j) in routes)
        objective_expr -= cost_expr

        for i in range(warehouses):
            model.addCons(quicksum(transport_flow_vars[(i, j)] for j in range(distribution_points)) == distribution_capacities[i], f"DistributionCapacity_{i}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 157,
        'edge_probability': 0.1,
        'affinity': 20,
        'graph_type': 'barabasi_albert',
        'max_capacity': 1138,
        'min_volume': 10000,
        'max_volume': 15000,
        'capacity_deviation_factor': 0.8,
        'warehouses': 750,
        'distribution_points': 84,
    }
    
    optimal_warehousing = OptimalWarehousingAndDistribution(parameters, seed=seed)
    instance = optimal_warehousing.generate_instance()
    solve_status, solve_time = optimal_warehousing.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")