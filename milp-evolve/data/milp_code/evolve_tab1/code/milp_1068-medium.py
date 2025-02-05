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

class UrbanParkPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_parks > 0 and self.n_plots > 0
        assert self.min_park_cost >= 0 and self.max_park_cost >= self.min_park_cost
        assert self.min_plot_cost >= 0 and self.max_plot_cost >= self.min_plot_cost
        assert self.min_park_area > 0 and self.max_park_area >= self.min_park_area
        assert self.min_energy_cost >= 0 and self.max_energy_cost >= self.min_energy_cost
        
        park_costs = np.random.randint(self.min_park_cost, self.max_park_cost + 1, self.n_parks)
        plot_costs = np.random.randint(self.min_plot_cost, self.max_plot_cost + 1, (self.n_parks, self.n_plots))
        areas = np.random.randint(self.min_park_area, self.max_park_area + 1, self.n_parks)
        demands = np.random.randint(1, 10, self.n_plots)
        energy_costs = np.random.uniform(self.min_energy_cost, self.max_energy_cost, self.n_parks)

        graph = Graph.barabasi_albert(self.n_parks, self.affinity)
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))
        
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(self.n_parks):
            if node not in used_nodes:
                inequalities.add((node,))
        
        # Generate network flow costs
        graph_flow = nx.erdos_renyi_graph(self.n_parks, 0.1, seed=self.seed)
        adj_matrix = nx.to_numpy_array(graph_flow)
        flow_costs = (adj_matrix * np.random.randint(1, 20, (self.n_parks, self.n_parks))).astype(int)
        
        return {
            "park_costs": park_costs,
            "plot_costs": plot_costs,
            "areas": areas,
            "demands": demands,
            "graph": graph,
            "inequalities": inequalities,
            "edge_weights": edge_weights,
            "flow_costs": flow_costs,
            "energy_costs": energy_costs,
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        park_costs = instance['park_costs']
        plot_costs = instance['plot_costs']
        areas = instance['areas']
        demands = instance['demands']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_weights = instance['edge_weights']
        flow_costs = instance['flow_costs']
        energy_costs = instance['energy_costs']
        
        model = Model("UrbanParkPlanning")
        n_parks = len(park_costs)
        n_plots = len(plot_costs[0])
        
        # Decision variables
        park_vars = {p: model.addVar(vtype="B", name=f"Park_{p}") for p in range(n_parks)}
        plot_vars = {(p, t): model.addVar(vtype="B", name=f"Park_{p}_Plot_{t}") for p in range(n_parks) for t in range(n_plots)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.edges}
        flow_vars = {(i, j): model.addVar(vtype="I", name=f"Flow_{i}_{j}") for i in range(n_parks) for j in range(n_parks)}
        multipurpose_park_vars = {p: model.addVar(vtype="B", name=f"Multipurpose_Park_{p}") for p in range(n_parks)}
        energy_cost_vars = {p: model.addVar(vtype="C", lb=0, name=f"ParkEnergy_{p}") for p in range(n_parks)}
        
        # Objective: minimize the total cost including flow costs, energy costs, and unused space penalties, while maximizing green cover
        model.setObjective(
            quicksum(park_costs[p] * park_vars[p] for p in range(n_parks)) +
            quicksum(plot_costs[p, t] * plot_vars[p, t] for p in range(n_parks) for t in range(n_plots)) +
            quicksum(edge_weights[i] * edge_vars[edge] for i, edge in enumerate(graph.edges)) +
            quicksum(flow_costs[i, j] * flow_vars[i, j] for i in range(n_parks) for j in range(n_parks)) +
            quicksum(energy_cost_vars[p] for p in range(n_parks)) -
            50 * quicksum(multipurpose_park_vars[p] for p in range(n_parks)), "minimize"
        )
        
        # Constraints: Each plot demand is met by exactly one park
        for t in range(n_plots):
            model.addCons(quicksum(plot_vars[p, t] for p in range(n_parks)) == 1, f"Plot_{t}_Demand")
        
        # Constraints: Only open parks can serve plots
        for p in range(n_parks):
            for t in range(n_plots):
                model.addCons(plot_vars[p, t] <= park_vars[p], f"Park_{p}_Serve_{t}")
        
        # Constraints: Parks cannot exceed their area
        for p in range(n_parks):
            model.addCons(quicksum(demands[t] * plot_vars[p, t] for t in range(n_plots)) <= areas[p], f"Park_{p}_Area")
        
        # Constraints: Park Graph Cliques
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(park_vars[node] for node in group) <= 1, f"Clique_{count}")

        # Neighboring Constraints: Prohibit neighboring parks being very similar
        for i, neighbors in graph.neighbors.items():
            for neighbor in neighbors:
                model.addCons(park_vars[i] + park_vars[neighbor] <= 1, f"Neighbor_{i}_{neighbor}")

        # Flow conservation constraints
        for i in range(n_parks):
            model.addCons(
                quicksum(flow_vars[i, j] for j in range(n_parks) if i != j) ==
                quicksum(flow_vars[j, i] for j in range(n_parks) if i != j),
                f"Flow_Conservation_{i}"
            )

        # Hysteresis delay to avoid rapid changes in park setups
        for p in range(n_parks):
            model.addCons(multipurpose_park_vars[p] <= park_vars[p], f"Multipurpose_{p}_Delay")

        # Energy cost constraints
        for p in range(n_parks):
            model.addCons(energy_cost_vars[p] == energy_costs[p] * areas[p] * park_vars[p], f"Park_{p}_EnergyCost")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_parks': 74,
        'n_plots': 112,
        'min_plot_cost': 7,
        'max_plot_cost': 3000,
        'min_park_cost': 1125,
        'max_park_cost': 5000,
        'min_park_area': 50,
        'max_park_area': 630,
        'affinity': 30,
        'flow_cost_range_min': 5,
        'flow_cost_range_max': 10,
        'min_energy_cost': 0.73,
        'max_energy_cost': 0.8,
    }

    park_planning_optimizer = UrbanParkPlanning(parameters, seed=42)
    instance = park_planning_optimizer.generate_instance()
    solve_status, solve_time, objective_value = park_planning_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")