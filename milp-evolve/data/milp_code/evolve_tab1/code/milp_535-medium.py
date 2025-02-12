import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WarehouseLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data Generation
    def generate_city_graph(self):
        # Generate a random city layout with fixed warehouses and clients
        G = nx.random_geometric_graph(self.n_total_nodes, self.geo_radius, seed=self.seed)
        adj_mat = np.zeros((self.n_total_nodes, self.n_total_nodes), dtype=object)
        edge_list = []
        warehouse_capacities = [random.randint(1, self.max_warehouse_capacity) for _ in range(self.n_warehouses)]
        client_demand = [random.randint(1, self.max_demand) for _ in range(self.n_clients)]

        for i, j in G.edges:
            cost = np.random.uniform(*self.service_cost_range)
            adj_mat[i, j] = cost
            edge_list.append((i, j))

        warehouses = range(self.n_warehouses)
        clients = range(self.n_clients, self.n_total_nodes)
        
        return G, adj_mat, edge_list, warehouse_capacities, client_demand, warehouses, clients

    def generate_instance(self):
        self.n_total_nodes = self.n_warehouses + self.n_clients
        G, adj_mat, edge_list, warehouse_capacities, client_demand, warehouses, clients = self.generate_city_graph()

        res = {
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'warehouse_capacities': warehouse_capacities, 
            'client_demand': client_demand, 
            'warehouses': warehouses, 
            'clients': clients
        }
        return res

    # PySCIPOpt Modeling
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        warehouse_capacities = instance['warehouse_capacities']
        client_demand = instance['client_demand']
        warehouses = instance['warehouses']
        clients = instance['clients']
        
        model = Model("WarehouseLocation")
        y_vars = {f"W_open_{w+1}": model.addVar(vtype="B", name=f"W_open_{w+1}") for w in warehouses}
        x_vars = {f"C_assign_{w+1}_{c+1}": model.addVar(vtype="B", name=f"C_assign_{w+1}_{c+1}") for w in warehouses for c in clients}
        
        # Objective function: minimize the total operation and service cost
        objective_expr = quicksum(
            adj_mat[w, c] * x_vars[f"C_assign_{w+1}_{c+1}"]
            for w in warehouses for c in clients
        )
        # Adding maintenance cost for opening a warehouse
        objective_expr += quicksum(
            self.operation_cost * y_vars[f"W_open_{w+1}"]
            for w in warehouses
        )

        model.setObjective(objective_expr, "minimize")

        # Constraints
        # Each client is served by at least one warehouse (Set Covering constraint)
        for c in clients:
            model.addCons(quicksum(x_vars[f"C_assign_{w+1}_{c+1}"] for w in warehouses) >= 1, f"Serve_{c+1}")

        # Warehouse should be open if it serves any client
        for w in warehouses:
            for c in clients:
                model.addCons(x_vars[f"C_assign_{w+1}_{c+1}"] <= y_vars[f"W_open_{w+1}"], f"Open_Cond_{w+1}_{c+1}")

        # Warehouse capacity constraint
        for w in warehouses:
            model.addCons(quicksum(client_demand[c-self.n_clients] * x_vars[f"C_assign_{w+1}_{c+1}"] for c in clients) <= warehouse_capacities[w], f"Capacity_{w+1}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 60,
        'n_clients': 2100,
        'service_cost_range': (400, 1600),
        'max_warehouse_capacity': 450,
        'max_demand': 100,
        'operation_cost': 5000,
        'geo_radius': 0.59,
    }
    warehouse_location = WarehouseLocation(parameters, seed=seed)
    instance = warehouse_location.generate_instance()
    solve_status, solve_time = warehouse_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")