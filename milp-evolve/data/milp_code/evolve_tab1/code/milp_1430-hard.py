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

    @staticmethod
    def erdos_renyi(number_of_nodes, probability, seed=None):
        graph = nx.erdos_renyi_graph(number_of_nodes, probability, seed=seed)
        edges = set(graph.edges)
        degrees = [d for (n, d) in graph.degree]
        neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes}
        return Graph(number_of_nodes, edges, degrees, neighbors)

class SimplifiedDataCenterResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        center_costs = np.random.randint(self.min_center_cost, self.max_center_cost + 1, self.n_centers)
        task_costs = np.random.randint(self.min_task_cost, self.max_task_cost + 1, (self.n_centers, self.n_tasks))
        capacities = np.random.randint(self.min_center_capacity, self.max_center_capacity + 1, self.n_centers)
        
        # Demands using normal distribution for more variation
        demands = np.random.normal(self.mean_task_demand, self.std_task_demand, self.n_tasks).astype(int) + 1

        graph = Graph.erdos_renyi(self.n_centers, self.link_probability, seed=self.seed)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))

        flow_costs = np.random.randint(1, 20, (self.n_centers, self.n_centers))
        flow_capacities = np.random.randint(self.flow_capacity_min, self.flow_capacity_max + 1, (self.n_centers, self.n_centers))
        
        # Transportation data
        raw_material_transport_costs = np.random.randint(10, 50, (self.n_centers, self.n_centers))

        return {
            "center_costs": center_costs,
            "task_costs": task_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "edge_weights": edge_weights,
            "flow_costs": flow_costs,
            "flow_capacities": flow_capacities,
            "raw_material_transport_costs": raw_material_transport_costs,
        }

    def solve(self, instance):
        center_costs = instance['center_costs']
        task_costs = instance['task_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        flow_costs = instance['flow_costs']
        flow_capacities = instance['flow_capacities']
        raw_material_transport_costs = instance['raw_material_transport_costs']

        model = Model("SimplifiedDataCenterResourceAllocation")
        n_centers = len(center_costs)
        n_tasks = len(task_costs[0])

        alloc_vars = {c: model.addVar(vtype="B", name=f"NetworkResource_Allocation_{c}") for c in range(n_centers)}
        task_vars = {(c, t): model.addVar(vtype="B", name=f"Center_{c}_Task_{t}") for c in range(n_centers) for t in range(n_tasks)}
        usage_vars = {(i, j): model.addVar(vtype="I", name=f"ComputationalResource_Usage_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}
        flow_vars = {(i, j): model.addVar(vtype="I", name=f"Flow_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}

        # Objective function
        model.setObjective(
            quicksum(center_costs[c] * alloc_vars[c] for c in range(n_centers)) +
            quicksum(task_costs[c, t] * task_vars[c, t] for c in range(n_centers) for t in range(n_tasks)) +
            quicksum(flow_costs[i, j] * flow_vars[i, j] for i in range(n_centers) for j in range(n_centers)) +
            quicksum(raw_material_transport_costs[i, j] * flow_vars[i, j] for i in range(n_centers) for j in range(n_centers)),
            "minimize"
        )

        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[c, t] for c in range(n_centers)) == 1, f"Task_{t}_Allocation")

        for c in range(n_centers):
            for t in range(n_tasks):
                model.addCons(task_vars[c, t] <= alloc_vars[c], f"Center_{c}_Serve_{t}")

        for c in range(n_centers):
            model.addCons(quicksum(demands[t] * task_vars[c, t] for t in range(n_tasks)) <= capacities[c], f"Center_{c}_Capacity")

        for edge in graph.edges:
            model.addCons(alloc_vars[edge[0]] + alloc_vars[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for i in range(n_centers):
            model.addCons(
                quicksum(usage_vars[i, j] for j in range(n_centers) if i != j) ==
                quicksum(usage_vars[j, i] for j in range(n_centers) if i != j),
                f"Usage_Conservation_{i}"
            )

        for j in range(n_centers):
            model.addCons(
                quicksum(flow_vars[i, j] for i in range(n_centers) if i != j) ==
                quicksum(task_vars[j, t] for t in range(n_tasks)),
                f"Flow_Conservation_{j}"
            )

        for i in range(n_centers):
            for j in range(n_centers):
                if i != j:
                    model.addCons(flow_vars[i, j] <= flow_capacities[i, j], f"Flow_Capacity_{i}_{j}")

        for c in range(n_centers):
            model.addCons(
                quicksum(task_vars[c, t] for t in range(n_tasks)) <= n_tasks * alloc_vars[c],
                f"Convex_Hull_{c}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_centers': 40,
        'n_tasks': 90,
        'min_task_cost': 800,
        'max_task_cost': 3000,
        'min_center_cost': 2000,
        'max_center_cost': 2000,
        'min_center_capacity': 400,
        'max_center_capacity': 1000,
        'link_probability': 0.45,
        'flow_capacity_min': 1000,
        'flow_capacity_max': 3000,
        'mean_task_demand': 6,
        'std_task_demand': 0.71,
    }

    resource_optimizer = SimplifiedDataCenterResourceAllocation(parameters, seed=seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")