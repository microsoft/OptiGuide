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
    def erdos_renyi(number_of_nodes, probability):
        G = nx.erdos_renyi_graph(n=number_of_nodes, p=probability)
        edges = set(G.edges)
        degrees = np.array([d for n, d in G.degree])
        neighbors = {node: set(G.neighbors(node)) for node in G.nodes}
        return Graph(number_of_nodes, edges, degrees, neighbors)

class LogisticsPackageRouting:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        assert self.n_nodes > 0 and self.n_customers > 0
        assert self.min_depot_cost >= 0 and self.max_depot_cost >= self.min_depot_cost
        assert self.min_customer_cost >= 0 and self.max_customer_cost >= self.min_customer_cost
        assert self.min_depot_capacity > 0 and self.max_depot_capacity >= self.min_depot_capacity

        depot_costs = np.random.randint(self.min_depot_cost, self.max_depot_cost + 1, self.n_nodes)
        customer_costs = np.random.randint(self.min_customer_cost, self.max_customer_cost + 1, (self.n_nodes, self.n_customers))
        capacities = np.random.randint(self.min_depot_capacity, self.max_depot_capacity + 1, self.n_nodes)
        demands = np.random.randint(1, 10, self.n_customers)

        graph = Graph.erdos_renyi(self.n_nodes, self.probability)

        return {
            "depot_costs": depot_costs,
            "customer_costs": customer_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
        }
        
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        depot_costs = instance['depot_costs']
        customer_costs = instance['customer_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        
        model = Model("LogisticsPackageRouting")
        n_nodes = len(depot_costs)
        n_customers = len(customer_costs[0])
        
        # Decision variables
        depot_open = {c: model.addVar(vtype="B", name=f"DepotOpen_{c}") for c in range(n_nodes)}
        customer_served = {(c, r): model.addVar(vtype="B", name=f"Depot_{c}_Customer_{r}") for c in range(n_nodes) for r in range(n_customers)}
        
        # Objective: minimize the total cost including depot costs and customer servicing costs
        model.setObjective(
            quicksum(depot_costs[c] * depot_open[c] for c in range(n_nodes)) +
            quicksum(customer_costs[c, r] * customer_served[c, r] for c in range(n_nodes) for r in range(n_customers)), "minimize"
        )
        
        # Constraints: Each customer is served by at least one depot
        for r in range(n_customers):
            model.addCons(quicksum(customer_served[c, r] for c in range(n_nodes)) >= 1, f"Customer_{r}_Service")
        
        # Constraints: Only open depots can serve customers
        for c in range(n_nodes):
            for r in range(n_customers):
                model.addCons(customer_served[c, r] <= depot_open[c], f"Depot_{c}_Serve_{r}")
        
        # Constraints: Depots cannot exceed their capacity
        for c in range(n_nodes):
            model.addCons(quicksum(demands[r] * customer_served[c, r] for r in range(n_customers)) <= capacities[c], f"Depot_{c}_Capacity")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 74,
        'n_customers': 56,
        'min_customer_cost': 1000,
        'max_customer_cost': 2500,
        'min_depot_cost': 2000,
        'max_depot_cost': 4000,
        'min_depot_capacity': 150,
        'max_depot_capacity': 1500,
        'probability': 0.73,
    }

    logistics_optimizer = LogisticsPackageRouting(parameters, seed)
    instance = logistics_optimizer.generate_instance()
    solve_status, solve_time, objective_value = logistics_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")