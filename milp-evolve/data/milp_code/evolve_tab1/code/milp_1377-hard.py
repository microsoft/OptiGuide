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

class WarehouseLogisticsOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.num_warehouses > 0 and self.num_retailers > 0
        assert self.min_warehouse_cost >= 0 and self.max_warehouse_cost >= self.min_warehouse_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_warehouse_capacity > 0 and self.max_warehouse_capacity >= self.min_warehouse_capacity

        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.num_warehouses)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.num_warehouses, self.num_retailers))
        capacities = np.random.randint(self.min_warehouse_capacity, self.max_warehouse_capacity + 1, self.num_warehouses)
        demands = np.random.randint(1, 20, self.num_retailers)

        # Generating delivery time windows, electricity consumption, and warehouse reliability data
        delivery_times = np.random.randint(1, 10, (self.num_warehouses, self.num_retailers))
        electricity_consumptions = np.random.randint(50, 500, (self.num_warehouses, self.num_retailers))
        warehouse_reliabilities = np.random.randint(0, 2, self.num_warehouses)

        graph = Graph.barabasi_albert(self.num_warehouses, self.affinity)
        cliques = []
        for clique in nx.find_cliques(nx.Graph(graph.edges)):
            if len(clique) > 1:
                cliques.append(tuple(sorted(clique)))

        return {
            "warehouse_costs": warehouse_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "demands": demands,
            "cliques": cliques,
            "delivery_times": delivery_times,
            "electricity_consumptions": electricity_consumptions,
            "warehouse_reliabilities": warehouse_reliabilities
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        warehouse_costs = instance['warehouse_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        cliques = instance['cliques']
        delivery_times = instance['delivery_times']
        electricity_consumptions = instance['electricity_consumptions']
        warehouse_reliabilities = instance['warehouse_reliabilities']
        
        model = Model("WarehouseLogisticsOptimization")
        num_warehouses = len(warehouse_costs)
        num_retailers = len(transport_costs[0])
        
        # Decision variables
        warehouse_open = {w: model.addVar(vtype="B", name=f"WarehouseOpen_{w}") for w in range(num_warehouses)}
        retailer_supplied = {(w, r): model.addVar(vtype="B", name=f"Warehouse_{w}_Retailer_{r}") for w in range(num_warehouses) for r in range(num_retailers)}
        electricity_use_vars = {(w, r): model.addVar(vtype="C", name=f"ElectricityUse_{w}_{r}") for w in range(num_warehouses) for r in range(num_retailers)}
        total_transport_costs = {r: model.addVar(vtype="C", name=f"TotalTransportCost_{r}") for r in range(num_retailers)}
        delivery_time_vars = {(w, r): model.addVar(vtype="C", name=f"DeliveryTime_{w}_{r}") for w in range(num_warehouses) for r in range(num_retailers)}
        reliability_vars = {w: model.addVar(vtype="B", name=f"WarehouseReliability_{w}") for w in range(num_warehouses)}

        # Objective: minimize total warehousing cost including warehouse setup, transportation, and electricity usage
        model.setObjective(
            quicksum(warehouse_costs[w] * warehouse_open[w] for w in range(num_warehouses)) +
            quicksum(transport_costs[w, r] * retailer_supplied[w, r] for w in range(num_warehouses) for r in range(num_retailers)) +
            quicksum(electricity_consumptions[w, r] * electricity_use_vars[w, r] for w in range(num_warehouses) for r in range(num_retailers)), "minimize"
        )
        
        # Constraints: Each retailer is supplied by at least one warehouse
        for r in range(num_retailers):
            model.addCons(quicksum(retailer_supplied[w, r] for w in range(num_warehouses)) >= 1, f"Retailer_{r}_Coverage")
        
        # Constraints: Only open warehouses can supply retailers
        for w in range(num_warehouses):
            for r in range(num_retailers):
                model.addCons(retailer_supplied[w, r] <= warehouse_open[w], f"Warehouse_{w}_Supply_{r}")
                model.addCons(electricity_use_vars[w, r] >= electricity_consumptions[w, r] * retailer_supplied[w, r], f"ElectricityConsumption_{w}_{r}")

        # Constraints: Warehouses cannot exceed their capacity
        for w in range(num_warehouses):
            model.addCons(quicksum(demands[r] * retailer_supplied[w, r] for r in range(num_retailers)) <= capacities[w], f"Warehouse_{w}_CapacityLimit")
        
        # Constraints: Warehouse Clique Limits
        for count, clique in enumerate(cliques):
            model.addCons(quicksum(warehouse_open[node] for node in clique) <= 1, f"WarehouseCliqueRestriction_{count}")

        # Constraints: Delivery time windows
        for w in range(num_warehouses):
            for r in range(num_retailers):
                model.addCons(delivery_time_vars[w, r] >= delivery_times[w, r] * retailer_supplied[w, r], f"DeliveryTime_{w}_{r}")

        # Constraints: Warehouse reliability
        for w in range(num_warehouses):
            model.addCons(reliability_vars[w] == warehouse_reliabilities[w], f"WarehouseReliability_{w}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_warehouses': 50,
        'num_retailers': 150,
        'min_transport_cost': 450,
        'max_transport_cost': 1500,
        'min_warehouse_cost': 750,
        'max_warehouse_cost': 1250,
        'min_warehouse_capacity': 25,
        'max_warehouse_capacity': 900,
        'affinity': 4,
    }

    warehouse_optimizer = WarehouseLogisticsOptimization(parameters, seed)
    instance = warehouse_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")