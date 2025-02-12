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

class ManufacturingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_machines > 0 and self.n_tasks > 0
        assert self.min_machine_cost >= 0 and self.max_machine_cost >= self.min_machine_cost
        assert self.min_task_cost >= 0 and self.max_task_cost >= self.min_task_cost
        assert self.min_machine_capacity > 0 and self.max_machine_capacity >= self.min_machine_capacity

        machine_costs = np.random.randint(self.min_machine_cost, self.max_machine_cost + 1, self.n_machines)
        task_costs = np.random.randint(self.min_task_cost, self.max_task_cost + 1, (self.n_machines, self.n_tasks))
        capacities = np.random.randint(self.min_machine_capacity, self.max_machine_capacity + 1, self.n_machines)
        demands = np.random.randint(1, 10, self.n_tasks)
        environmental_impacts = np.random.uniform(self.min_env_impact, self.max_env_impact, (self.n_machines, self.n_machines))
        
        labor_costs = np.random.randint(self.min_labor_cost, self.max_labor_cost + 1, self.n_tasks)
        raw_material_costs = np.random.randint(self.min_raw_material_cost, self.max_raw_material_cost + 1, self.n_tasks)
        waste_limits = np.random.randint(self.min_waste_limit, self.max_waste_limit + 1, self.n_machines)
        
        graph = Graph.barabasi_albert(self.n_machines, self.affinity)
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
        for node in range(self.n_machines):
            if node not in used_nodes:
                inequalities.add((node,))

        # Simulating neighborhoods with socioeconomic weights and available land areas.
        socioeconomic_weights = np.random.uniform(1, 10, self.n_neighborhoods)
        land_areas = np.random.uniform(self.min_land_area, self.max_land_area, self.n_neighborhoods)
        
        return {
            "machine_costs": machine_costs,
            "task_costs": task_costs,
            "capacities": capacities,
            "demands": demands,
            "environmental_impacts": environmental_impacts,
            "graph": graph,
            "inequalities": inequalities,
            "edge_weights": edge_weights,
            "labor_costs": labor_costs,
            "raw_material_costs": raw_material_costs,
            "waste_limits": waste_limits,
            "socioeconomic_weights": socioeconomic_weights,
            "land_areas": land_areas
        }

    def solve(self, instance):
        machine_costs = instance['machine_costs']
        task_costs = instance['task_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        environmental_impacts = instance['environmental_impacts']
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_weights = instance['edge_weights']
        labor_costs = instance['labor_costs']
        raw_material_costs = instance['raw_material_costs']
        waste_limits = instance['waste_limits']
        socioeconomic_weights = instance['socioeconomic_weights']
        land_areas = instance['land_areas']

        model = Model("ManufacturingOptimization")
        n_machines = len(machine_costs)
        n_tasks = len(task_costs[0])
        n_neighborhoods = len(socioeconomic_weights)

        machine_vars = {m: model.addVar(vtype="B", name=f"Machine_{m}") for m in range(n_machines)}
        task_vars = {(m, t): model.addVar(vtype="B", name=f"Machine_{m}_Task_{t}") for m in range(n_machines) for t in range(n_tasks)}
        edge_vars = {edge: model.addVar(vtype="B", name=f"Edge_{edge[0]}_{edge[1]}") for edge in graph.edges}
        env_impact_vars = {(i, j): model.addVar(vtype="C", name=f"EnvImpact_{i}_{j}") for i in range(n_machines) for j in range(n_machines)}

        # New continuous variables for waste emissions
        waste_emission_vars = {(m): model.addVar(vtype="C", name=f"WasteEmission_{m}") for m in range(n_machines)}
        
        # New variables for neighborhood access and land usage
        access_vars = {n: model.addVar(vtype="B", name=f"Access_{n}") for n in range(n_neighborhoods)}
        land_usage_vars = {m: model.addVar(vtype="C", name=f"LandUsage_{m}") for m in range(n_machines)}

        model.setObjective(
            quicksum(machine_costs[m] * machine_vars[m] for m in range(n_machines)) +
            quicksum(task_costs[m, t] * task_vars[m, t] for m in range(n_machines) for t in range(n_tasks)) +
            quicksum(edge_weights[i] * edge_vars[edge] for i, edge in enumerate(graph.edges)) +
            quicksum(labor_costs[t] * task_vars[m, t] for m in range(n_machines) for t in range(n_tasks)) +
            quicksum(raw_material_costs[t] * task_vars[m, t] for m in range(n_machines) for t in range(n_tasks)) +
            1000 * quicksum(waste_emission_vars[m] for m in range(n_machines)) + 
            quicksum(1e5 * (1 - access_vars[n]) * socioeconomic_weights[n] for n in range(n_neighborhoods)) +  # Penalty for not covering neighborhoods
            quicksum(100 * land_usage_vars[m] for m in range(n_machines)), "minimize"
        )

        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[m, t] for m in range(n_machines)) == 1, f"Task_{t}_Demand")

        for m in range(n_machines):
            for t in range(n_tasks):
                model.addCons(task_vars[m, t] <= machine_vars[m], f"Machine_{m}_Serve_{t}")

        for m in range(n_machines):
            model.addCons(quicksum(demands[t] * task_vars[m, t] for t in range(n_tasks)) <= capacities[m], f"Machine_{m}_Capacity")

        for count, group in enumerate(inequalities):
            model.addCons(quicksum(machine_vars[node] for node in group) <= 1, f"Clique_{count}")

        for i, neighbors in graph.neighbors.items():
            for neighbor in neighbors:
                model.addCons(machine_vars[i] + machine_vars[neighbor] <= 1, f"Neighbor_{i}_{neighbor}")

        for i in range(n_machines):
            for j in range(i + 1, n_machines):
                model.addCons(env_impact_vars[i, j] >= (machine_vars[i] + machine_vars[j] - 1) * environmental_impacts[i, j], f"EnvImpact_{i}_{j}_Lower")
                model.addCons(env_impact_vars[i, j] <= machine_vars[i] * environmental_impacts[i, j], f"EnvImpact_{i}_{j}_Upperi")
                model.addCons(env_impact_vars[i, j] <= machine_vars[j] * environmental_impacts[i, j], f"EnvImpact_{i}_{j}_Upperj")

        for m in range(n_machines):
            model.addCons(waste_emission_vars[m] <= waste_limits[m], f"WasteLimit_{m}")
            
            # Land usage constraints
            model.addCons(land_usage_vars[m] <= machine_vars[m] * land_areas[m % n_neighborhoods], f"LandUsage_{m}")

        # Ensure at least one machine is deployed in each neighborhood based on socioeconomic weights
        for n in range(n_neighborhoods):
            model.addCons(quicksum(machine_vars[m] for m in range(n_machines) if m % n_neighborhoods == n) >= access_vars[n], f"NeighborhoodAccess_{n}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_machines': 50,
        'n_tasks': 120,
        'min_task_cost': 150,
        'max_task_cost': 2000,
        'min_machine_cost': 2400,
        'max_machine_cost': 3000,
        'min_machine_capacity': 200,
        'max_machine_capacity': 1000,
        'affinity': 15,
        'min_env_impact': 0.73,
        'max_env_impact': 500.0,
        'min_labor_cost': 30,
        'max_labor_cost': 50,
        'min_raw_material_cost': 40,
        'max_raw_material_cost': 600,
        'min_waste_limit': 75,
        'max_waste_limit': 450,
        'n_neighborhoods': 5,
        'min_land_area': 75,
        'max_land_area': 375,
    }

    manufacturing_optimizer = ManufacturingOptimization(parameters, seed=42)
    instance = manufacturing_optimizer.generate_instance()
    solve_status, solve_time, objective_value = manufacturing_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")