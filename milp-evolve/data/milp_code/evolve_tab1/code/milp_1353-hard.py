import random
import time
import numpy as np
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

class DistributedManufacturingNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        factory_costs = np.random.randint(self.min_factory_cost, self.max_factory_cost + 1, self.n_factories)
        product_costs = np.random.randint(self.min_product_cost, self.max_product_cost + 1, (self.n_factories, self.n_products))
        capacities = np.random.randint(self.min_factory_capacity, self.max_factory_capacity + 1, self.n_factories)
        demands = np.random.randint(1, 10, self.n_products)

        graph = Graph.erdos_renyi(self.n_factories, self.link_probability, seed=self.seed)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))

        flow_costs = np.random.randint(1, 20, (self.n_factories, self.n_factories))
        flow_capacities = np.random.randint(self.flow_capacity_min, self.flow_capacity_max + 1, (self.n_factories, self.n_factories))
        
        # Maintenance costs
        maintenance_costs = np.random.randint(5, 50, self.n_factories)
        
        # Emission costs
        emission_costs = np.random.randint(5, 30, (self.n_factories, self.n_factories))
        
        return {
            "factory_costs": factory_costs,
            "product_costs": product_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "edge_weights": edge_weights,
            "flow_costs": flow_costs,
            "flow_capacities": flow_capacities,
            "maintenance_costs": maintenance_costs,
            "emission_costs": emission_costs,
        }

    def solve(self, instance):
        factory_costs = instance['factory_costs']
        product_costs = instance['product_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        flow_costs = instance['flow_costs']
        flow_capacities = instance['flow_capacities']
        maintenance_costs = instance['maintenance_costs']
        emission_costs = instance['emission_costs']

        model = Model("DistributedManufacturingNetwork")
        n_factories = len(factory_costs)
        n_products = len(product_costs[0])

        activate_vars = {f: model.addVar(vtype="B", name=f"FactoryActivation_{f}") for f in range(n_factories)}
        production_vars = {(f, p): model.addVar(vtype="B", name=f"Factory_{f}_Product_{p}") for f in range(n_factories) for p in range(n_products)}
        usage_vars = {(i, j): model.addVar(vtype="I", name=f"MachineUsage_{i}_{j}") for i in range(n_factories) for j in range(n_factories)}
        flow_vars = {(i, j): model.addVar(vtype="I", name=f"ProductFlow_{i}_{j}") for i in range(n_factories) for j in range(n_factories)}

        # Objective function
        model.setObjective(
            quicksum(factory_costs[f] * activate_vars[f] for f in range(n_factories)) +
            quicksum(product_costs[f, p] * production_vars[f, p] for f in range(n_factories) for p in range(n_products)) +
            quicksum(flow_costs[i, j] * flow_vars[i, j] for i in range(n_factories) for j in range(n_factories)) +
            quicksum(maintenance_costs[f] * activate_vars[f] for f in range(n_factories)) +
            quicksum(emission_costs[i, j] * flow_vars[i, j] for i in range(n_factories) for j in range(n_factories)),
            "minimize"
        )

        for p in range(n_products):
            model.addCons(quicksum(production_vars[f, p] for f in range(n_factories)) == 1, f"Product_{p}_Production")

        for f in range(n_factories):
            for p in range(n_products):
                model.addCons(production_vars[f, p] <= activate_vars[f], f"Factory_{f}_Produces_{p}")

        for f in range(n_factories):
            model.addCons(quicksum(demands[p] * production_vars[f, p] for p in range(n_products)) <= capacities[f], f"Factory_{f}_Capacity")

        for edge in graph.edges:
            model.addCons(activate_vars[edge[0]] + activate_vars[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for i in range(n_factories):
            model.addCons(
                quicksum(usage_vars[i, j] for j in range(n_factories) if i != j) ==
                quicksum(usage_vars[j, i] for j in range(n_factories) if i != j),
                f"Usage_Conservation_{i}"
            )

        for j in range(n_factories):
            model.addCons(
                quicksum(flow_vars[i, j] for i in range(n_factories) if i != j) ==
                quicksum(production_vars[j, p] for p in range(n_products)),
                f"Flow_Conservation_{j}"
            )

        for i in range(n_factories):
            for j in range(n_factories):
                if i != j:
                    model.addCons(flow_vars[i, j] <= flow_capacities[i, j], f"Flow_Capacity_{i}_{j}")

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_factories': 36,
        'n_products': 100,
        'min_product_cost': 1000,
        'max_product_cost': 3000,
        'min_factory_cost': 2000,
        'max_factory_cost': 5000,
        'min_factory_capacity': 600,
        'max_factory_capacity': 1500,
        'link_probability': 0.8,
        'flow_capacity_min': 2000,
        'flow_capacity_max': 5000,
    }

    manufacturing_optimizer = DistributedManufacturingNetwork(parameters, seed=seed)
    instance = manufacturing_optimizer.generate_instance()
    solve_status, solve_time, objective_value = manufacturing_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")