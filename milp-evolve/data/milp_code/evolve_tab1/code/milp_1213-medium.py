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

class DataCenterResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_centers > 0 and self.n_tasks > 0
        assert self.min_center_cost >= 0 and self.max_center_cost >= self.min_center_cost
        assert self.min_task_cost >= 0 and self.max_task_cost >= self.min_task_cost
        assert self.min_center_capacity > 0 and self.max_center_capacity >= self.min_center_capacity

        center_costs = np.random.randint(self.min_center_cost, self.max_center_cost + 1, self.n_centers)
        task_costs = np.random.randint(self.min_task_cost, self.max_task_cost + 1, (self.n_centers, self.n_tasks))
        capacities = np.random.randint(self.min_center_capacity, self.max_center_capacity + 1, self.n_centers)
        demands = np.random.randint(1, 10, self.n_tasks)
        migration_costs = np.random.randint(self.min_migration_cost, self.max_migration_cost + 1, self.n_centers)
        hardware_health = np.random.uniform(self.min_hw_health, self.max_hw_health, (self.n_centers, self.n_centers))

        graph = Graph.barabasi_albert(self.n_centers, self.affinity)
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
        for node in range(self.n_centers):
            if node not in used_nodes:
                inequalities.add((node,))

        graph_flow = nx.erdos_renyi_graph(self.n_centers, 0.1, seed=self.seed)
        adj_matrix = nx.to_numpy_array(graph_flow)
        flow_costs = (adj_matrix * np.random.randint(1, 20, (self.n_centers, self.n_centers))).astype(int)

        performance_impact = np.random.normal(self.performance_mean, self.performance_std, self.n_centers)

        flow_capacities = np.random.randint(self.flow_capacity_min, self.flow_capacity_max + 1, (self.n_centers, self.n_centers))

        return {
            "center_costs": center_costs,
            "task_costs": task_costs,
            "capacities": capacities,
            "demands": demands,
            "migration_costs": migration_costs,
            "hardware_health": hardware_health,
            "graph": graph,
            "inequalities": inequalities,
            "edge_weights": edge_weights,
            "flow_costs": flow_costs,
            "performance_impact": performance_impact,
            "flow_capacities": flow_capacities,
            ### new instance data code ends here
            "node_weights": np.random.randint(1, 10, size=self.n_centers),  # New data: node weights for independent set constraints
            "capacities_bin": np.random.randint(10, 20, size=self.n_centers),  # New data: bin capacities
        }

    def solve(self, instance):
        center_costs = instance['center_costs']
        task_costs = instance['task_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        migration_costs = instance['migration_costs']
        hardware_health = instance['hardware_health']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_weights = instance['edge_weights']
        flow_costs = instance['flow_costs']
        performance_impact = instance['performance_impact']
        flow_capacities = instance['flow_capacities']

        model = Model("DataCenterResourceAllocation")
        n_centers = len(center_costs)
        n_tasks = len(task_costs[0])

        alloc_vars = {c: model.addVar(vtype="B", name=f"NetworkResource_Allocation_{c}") for c in range(n_centers)}
        task_vars = {(c, t): model.addVar(vtype="B", name=f"Center_{c}_Task_{t}") for c in range(n_centers) for t in range(n_tasks)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.edges}
        usage_vars = {(i, j): model.addVar(vtype="I", name=f"ComputationalResource_Usage_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}
        multi_usage_vars = {c: model.addVar(vtype="B", name=f"MaximumCapacity_{c}") for c in range(n_centers)}
        migration_vars = {c: model.addVar(vtype="I", name=f"MigrationCosts_{c}") for c in range(n_centers)}
        hw_health_vars = {(i, j): model.addVar(vtype="C", name=f"HardwareHealth_Status_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}

        # Performance impact variables
        performance_impact_vars = {c: model.addVar(vtype="C", name=f"PerformanceImpact_{c}") for c in range(n_centers)}
        
        # New flow variables
        flow_vars = {(i, j): model.addVar(vtype="I", name=f"Flow_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}

        ### Add new constraint variables for independent set and bin packing
        node_weight = instance["node_weights"]
        capacities_bin = instance["capacities_bin"]
        independent_vars = {c: model.addVar(vtype="B", name=f"Independent_Variables_{c}") for c in range(n_centers)}

        ### Objective Function
        model.setObjective(
            quicksum(center_costs[c] * alloc_vars[c] for c in range(n_centers)) +
            quicksum(task_costs[c, t] * task_vars[c, t] for c in range(n_centers) for t in range(n_tasks)) +
            quicksum(edge_weights[i] * edge_vars[edge] for i, edge in enumerate(graph.edges)) +
            quicksum(flow_costs[i, j] * usage_vars[i, j] for i in range(n_centers) for j in range(n_centers)) +
            quicksum(migration_costs[c] * migration_vars[c] for c in range(n_centers)) +
            quicksum(hardware_health[i, j] * hw_health_vars[i, j] for i in range(n_centers) for j in range(n_centers)) -
            50 * quicksum(performance_impact[c] * performance_impact_vars[c] for c in range(n_centers)) +
            quicksum(flow_costs[i, j] * flow_vars[i, j] for i in range(n_centers) for j in range(n_centers)), "minimize"
        )

        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[c, t] for c in range(n_centers)) == 1, f"Task_{t}_Allocation")

        for c in range(n_centers):
            for t in range(n_tasks):
                model.addCons(task_vars[c, t] <= alloc_vars[c], f"Center_{c}_Serve_{t}")

        for c in range(n_centers):
            model.addCons(quicksum(demands[t] * task_vars[c, t] for t in range(n_tasks)) <= capacities[c], f"Center_{c}_Capacity")

        for count, group in enumerate(inequalities):
            model.addCons(quicksum(alloc_vars[node] for node in group) <= 1, f"Clique_{count}")

        for i, neighbors in graph.neighbors.items():
            for neighbor in neighbors:
                model.addCons(alloc_vars[i] + alloc_vars[neighbor] <= 1, f"Neighbor_{i}_{neighbor}")

        for i in range(n_centers):
            model.addCons(
                quicksum(usage_vars[i, j] for j in range(n_centers) if i != j) ==
                quicksum(usage_vars[j, i] for j in range(n_centers) if i != j),
                f"Usage_Conservation_{i}"
            )

        for c in range(n_centers):
            model.addCons(multi_usage_vars[c] <= alloc_vars[c], f"MaximumCapacity_{c}_Delay")

        # Performance impact constraints
        for c in range(n_centers):
            model.addCons(performance_impact_vars[c] == alloc_vars[c] * performance_impact[c], f"PerformanceImpact_{c}")

        # Indicator constraints for resource usage
        for i in range(n_centers):
            for j in range(n_centers):
                model.addCons(usage_vars[i, j] >= self.min_usage_amount * alloc_vars[i] * alloc_vars[j], f"Usage_{i}_{j}_MinUsage")
                model.addCons(usage_vars[i, j] <= self.max_usage_amount * alloc_vars[i] * alloc_vars[j], f"Usage_{i}_{j}_MaxUsage")

        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                model.addCons(hw_health_vars[i, j] >= alloc_vars[i] * alloc_vars[j] * hardware_health[i, j], f"HardwareHealth_{i}_{j}")

        # Flow conservation constraints
        for j in range(n_centers):
            model.addCons(
                quicksum(flow_vars[i, j] for i in range(n_centers) if i != j) ==
                quicksum(task_vars[j, t] for t in range(n_tasks)),
                f"Flow_Conservation_{j}"
            )

        # Flow capacity constraints
        for i in range(n_centers):
            for j in range(n_centers):
                if i != j:
                    model.addCons(flow_vars[i, j] <= flow_capacities[i, j], f"Flow_Capacity_{i}_{j}")

        # Independent set constraints
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(independent_vars[node] for node in group) <= 1, f"Independent_Set_{count}")

        # Bin packing constraints
        for i in range(len(capacities_bin)):
            model.addCons(quicksum(node_weight[node] * independent_vars[node] for node in range(i, len(independent_vars), len(capacities_bin))) <= capacities_bin[i], f"Bin_Packing_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_centers': 55,
        'n_tasks': 84,
        'min_task_cost': 551,
        'max_task_cost': 3000,
        'min_center_cost': 632,
        'max_center_cost': 5000,
        'min_center_capacity': 423,
        'max_center_capacity': 1012,
        'affinity': 1,
        'flow_cost_range_min': 99,
        'flow_cost_range_max': 675,
        'min_migration_cost': 525,
        'max_migration_cost': 3000,
        'min_hw_health': 0.45,
        'max_hw_health': 675.0,
        'min_usage_amount': 750,
        'max_usage_amount': 750,
        'performance_mean': 0.59,
        'performance_std': 0.38,
        'flow_capacity_min': 500,
        'flow_capacity_max': 3000,
        ### new parameter code ends here
        'graph_type': 'barabasi_albert',  # Added a graph type parameter for consistency with the second MILP data generation method
    }

    resource_optimizer = DataCenterResourceAllocation(parameters, seed=seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")