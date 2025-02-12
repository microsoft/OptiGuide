import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

# Define the Graph class for adjacency modeling
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

class SupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_warehouses > 0 and self.n_stores > 0
        assert self.min_warehouse_cost >= 0 and self.max_warehouse_cost >= self.min_warehouse_cost
        assert self.min_store_cost >= 0 and self.max_store_cost >= self.min_store_cost
        assert self.min_warehouse_cap > 0 and self.max_warehouse_cap >= self.min_warehouse_cap

        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        store_costs = np.random.randint(self.min_store_cost, self.max_store_cost + 1, (self.n_warehouses, self.n_stores))
        capacities = np.random.randint(self.min_warehouse_cap, self.max_warehouse_cap + 1, self.n_warehouses)
        demands = np.random.randint(1, 10, self.n_stores)
        supply_limits = np.random.uniform(self.min_supply_limit, self.max_supply_limit, self.n_warehouses)
        distances = np.random.uniform(0, self.max_distance, (self.n_warehouses, self.n_stores))
        
        # Generate graph for adjacency constraints
        graph = Graph.barabasi_albert(self.n_warehouses, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()

        return {
            "warehouse_costs": warehouse_costs,
            "store_costs": store_costs,
            "capacities": capacities,
            "demands": demands,
            "supply_limits": supply_limits,
            "distances": distances,
            "graph": graph,
            "cliques": cliques,
        }

    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        store_costs = instance['store_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        supply_limits = instance['supply_limits']
        distances = instance['distances']
        graph = instance['graph']
        cliques = instance['cliques']

        model = Model("SupplyChainOptimization")
        n_warehouses = len(warehouse_costs)
        n_stores = len(store_costs[0])

        # Decision variables
        open_vars = {w: model.addVar(vtype="B", name=f"Warehouse_{w}") for w in range(n_warehouses)}
        flow_vars = {(w, s): model.addVar(vtype="C", name=f"Flow_{w}_{s}") for w in range(n_warehouses) for s in range(n_stores)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.edges}

        # Objective: minimize the total cost including warehouse costs, store costs, and edge weights.
        model.setObjective(
            quicksum(warehouse_costs[w] * open_vars[w] for w in range(n_warehouses)) +
            quicksum(store_costs[w, s] * flow_vars[w, s] for w in range(n_warehouses) for s in range(n_stores)),
            "minimize"
        )

        # Constraints: Each store's demand is met by the warehouses
        for s in range(n_stores):
            model.addCons(quicksum(flow_vars[w, s] for w in range(n_warehouses)) == demands[s], f"Store_{s}_Demand")
        
        # Constraints: Only open warehouses can supply products
        for w in range(n_warehouses):
            for s in range(n_stores):
                model.addCons(flow_vars[w, s] <= supply_limits[w] * open_vars[w], f"Warehouse_{w}_Serve_{s}")
        
        # Constraints: Warehouses cannot exceed their capacities
        for w in range(n_warehouses):
            model.addCons(quicksum(flow_vars[w, s] for s in range(n_stores)) <= capacities[w], f"Warehouse_{w}_Capacity")

        # New constraint: Service distance constraint (Set Covering)
        for s in range(n_stores):
            model.addCons(quicksum(open_vars[w] for w in range(n_warehouses) if distances[w, s] <= self.max_distance) >= 1, f"Store_{s}_Covered")

        # Adjacency constraints based on cliques
        for count, clique in enumerate(cliques):
            model.addCons(quicksum(open_vars[node] for node in clique) <= 1, f"Clique_{count}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 87,
        'n_stores': 75,
        'min_store_cost': 600,
        'max_store_cost': 3000,
        'min_warehouse_cost': 3000,
        'max_warehouse_cost': 3000,
        'min_warehouse_cap': 2000,
        'max_warehouse_cap': 3000,
        'min_supply_limit': 1050,
        'max_supply_limit': 1125,
        'max_distance': 150,
        'affinity': 77,
    }

    supply_optimizer = SupplyChainOptimization(parameters, seed=seed)
    instance = supply_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")