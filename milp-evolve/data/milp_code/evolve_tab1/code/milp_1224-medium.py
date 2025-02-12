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

        # Generating stochastic data for climate-related variables
        precipitation = np.random.normal(self.precipitation_mean, self.precipitation_std, self.n_centers)
        temperature = np.random.normal(self.temperature_mean, self.temperature_std, self.n_centers)
        extreme_weather_prob = np.random.uniform(0, 1, self.n_centers)

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
            "precipitation": precipitation,
            "temperature": temperature,
            "extreme_weather_prob": extreme_weather_prob
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
        precipitation = instance['precipitation']
        temperature = instance['temperature']
        extreme_weather_prob = instance['extreme_weather_prob']

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

        # Climate impact variables
        climate_impact_vars = {c: model.addVar(vtype="B", name=f"ClimateImpactVar_{c}") for c in range(n_centers)}
        climatic_delay_vars = {c: model.addVar(vtype="C", name=f"ClimaticDelayVar_{c}") for c in range(n_centers)}

        # Objective Function
        model.setObjective(
            quicksum(center_costs[c] * alloc_vars[c] for c in range(n_centers)) +
            quicksum(task_costs[c, t] * task_vars[c, t] for c in range(n_centers) for t in range(n_tasks)) +
            quicksum(edge_weights[i] * edge_vars[edge] for i, edge in enumerate(graph.edges)) +
            quicksum(flow_costs[i, j] * usage_vars[i, j] for i in range(n_centers) for j in range(n_centers)) +
            quicksum(migration_costs[c] * migration_vars[c] for c in range(n_centers)) +
            quicksum(hardware_health[i, j] * hw_health_vars[i, j] for i in range(n_centers) for j in range(n_centers)) -
            50 * quicksum(performance_impact[c] * performance_impact_vars[c] for c in range(n_centers)) +
            quicksum(self.climatic_cost_factor * climatic_delay_vars[c] for c in range(n_centers)) -
            100 * quicksum(climate_impact_vars[c] * extreme_weather_prob[c] for c in range(n_centers)), "minimize"
        )

        # Task Assignment Constraints
        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[c, t] for c in range(n_centers)) == 1, f"Task_{t}_Allocation")

        for c in range(n_centers):
            for t in range(n_tasks):
                model.addCons(task_vars[c, t] <= alloc_vars[c], f"Center_{c}_Serve_{t}")

        for c in range(n_centers):
            model.addCons(quicksum(demands[t] * task_vars[c, t] for t in range(n_tasks)) <= capacities[c], f"Center_{c}_Capacity")

        # Clique Constraints
        for count, group in enumerate(inequalities):
            model.addCons(quicksum(alloc_vars[node] for node in group) <= 1, f"Clique_{count}")

        # Neighbor Constraints Removal and adding modified Clique Constraints
        for i in range(n_centers):
            neighbors = graph.neighbors[i]
            model.addCons(quicksum(alloc_vars[j] for j in neighbors) <= (1 - alloc_vars[i]) + len(neighbors) * (1 - alloc_vars[i]), f"ModifiedClique_{i}")

        # Flow Usage Constraints
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

        # Climate impact constraints
        for c in range(n_centers):
            model.addCons(climate_impact_vars[c] <= extreme_weather_prob[c], f"ClimateImpact_{c}_ExtremeWeather")

        for c in range(n_centers):
            model.addCons(climatic_delay_vars[c] >= alloc_vars[c] * precipitation[c], f"ClimaticDelay_{c}_Precipitation")
            model.addCons(climatic_delay_vars[c] <= alloc_vars[c] * temperature[c], f"ClimaticDelay_{c}_Temperature")

        model.addCons(quicksum(climate_impact_vars[c] for c in range(n_centers)) <= self.max_total_climate_impact, "OverallClimateImpact")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_centers': 27,
        'n_tasks': 280,
        'min_task_cost': 2205,
        'max_task_cost': 3000,
        'min_center_cost': 84,
        'max_center_cost': 5000,
        'min_center_capacity': 282,
        'max_center_capacity': 1012,
        'affinity': 1,
        'flow_cost_range_min': 1,
        'flow_cost_range_max': 3,
        'min_migration_cost': 262,
        'max_migration_cost': 3000,
        'min_hw_health': 0.73,
        'max_hw_health': 37.5,
        'min_usage_amount': 750,
        'max_usage_amount': 2500,
        'performance_mean': 0.59,
        'performance_std': 0.59,
        'precipitation_mean': 3.75,
        'precipitation_std': 2.0,
        'temperature_mean': 16.5,
        'temperature_std': 50.0,
        'climatic_cost_factor': 1.0,
        'max_total_climate_impact': 2,
    }
    resource_optimizer = DataCenterResourceAllocation(parameters, seed=42)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")