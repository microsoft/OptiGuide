import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_devices > 0 and self.n_data_requests > 0
        assert self.min_cost_device >= 0 and self.max_cost_device >= self.min_cost_device
        assert self.min_cost_connection >= 0 and self.max_cost_connection >= self.min_cost_connection
        assert self.min_capacity_device > 0 and self.max_capacity_device >= self.min_capacity_device
        assert self.min_data_demand >= 0 and self.max_data_demand >= self.min_data_demand

        device_operational_costs = np.random.randint(self.min_cost_device, self.max_cost_device + 1, self.n_devices)
        connection_costs = np.random.randint(self.min_cost_connection, self.max_cost_connection + 1, (self.n_devices, self.n_data_requests))
        device_capacities = np.random.randint(self.min_capacity_device, self.max_capacity_device + 1, self.n_devices)
        data_demands = np.random.randint(self.min_data_demand, self.max_data_demand + 1, self.n_data_requests)
        no_connection_penalties = np.random.uniform(100, 300, self.n_data_requests).tolist()

        # Generate network graph
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        G, adj_mat, edge_list = self.generate_erdos_graph()
        network_costs, network_set = self.generate_network_data()
        
        return {
            "device_operational_costs": device_operational_costs,
            "connection_costs": connection_costs,
            "device_capacities": device_capacities,
            "data_demands": data_demands,
            "no_connection_penalties": no_connection_penalties,
            "adj_mat": adj_mat,
            "edge_list": edge_list,
            "network_costs": network_costs,
            "network_set": network_set
        }

    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))

        return G, adj_mat, edge_list
    
    def generate_network_data(self):
        network_costs = np.random.randint(1, self.network_max_cost + 1, size=self.n_nodes)
        network_pairs = [(i, j) for i in range(self.n_nodes) for j in range(self.n_nodes) if i != j]
        chosen_pairs = np.random.choice(len(network_pairs), size=self.n_network_pairs, replace=False)
        network_set = [network_pairs[i] for i in chosen_pairs]
        return network_costs, network_set

    def solve(self, instance):
        device_operational_costs = instance['device_operational_costs']
        connection_costs = instance['connection_costs']
        device_capacities = instance['device_capacities']
        data_demands = instance['data_demands']
        no_connection_penalties = instance['no_connection_penalties']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        network_costs = instance['network_costs']
        network_set = instance['network_set']

        model = Model("NetworkOptimization")
        n_devices = len(device_operational_costs)
        n_data_requests = len(data_demands)
        
        device_vars = {d: model.addVar(vtype="B", name=f"Device_{d}") for d in range(n_devices)}
        data_transfer_vars = {(d, r): model.addVar(vtype="C", name=f"Data_{d}_Request_{r}") for d in range(n_devices) for r in range(n_data_requests)}
        unmet_data_vars = {r: model.addVar(vtype="C", name=f"Unmet_Data_{r}") for r in range(n_data_requests)}

        # New Variables: Network usage and hazard handling variables
        network_vars = {(i,j): model.addVar(vtype="B", name=f"Network_{i}_{j}") for (i,j) in edge_list}
        hazard_vars = {(i, j): model.addVar(vtype="B", name=f"Hazard_{i}_{j}") for (i, j) in network_set}

        model.setObjective(
            quicksum(device_operational_costs[d] * device_vars[d] for d in range(n_devices)) +
            quicksum(connection_costs[d][r] * data_transfer_vars[d, r] for d in range(n_devices) for r in range(n_data_requests)) +
            quicksum(no_connection_penalties[r] * unmet_data_vars[r] for r in range(n_data_requests)) +
            quicksum(adj_mat[i, j][1] * network_vars[(i,j)] for (i,j) in edge_list) +
            quicksum(network_costs[i] * hazard_vars[(i, j)] for (i, j) in network_set),
            "minimize"
        )

        # Constraints
        # Data demand satisfaction
        for r in range(n_data_requests):
            model.addCons(quicksum(data_transfer_vars[d, r] for d in range(n_devices)) + unmet_data_vars[r] == data_demands[r], f"Data_Demand_Satisfaction_{r}")
        
        # Capacity limits for each device
        for d in range(n_devices):
            model.addCons(quicksum(data_transfer_vars[d, r] for r in range(n_data_requests)) <= device_capacities[d] * device_vars[d], f"Device_Capacity_{d}")

        # Data transfer only if device is operational
        for d in range(n_devices):
            for r in range(n_data_requests):
                model.addCons(data_transfer_vars[d, r] <= data_demands[r] * device_vars[d], f"Operational_Constraint_{d}_{r}")

        # Hazard handling constraints
        for i, j in network_set:
            model.addCons(hazard_vars[(i, j)] == 1, f"Hazard_Handling_{i}_{j}")
 
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_devices': 200,
        'n_data_requests': 600,
        'min_cost_device': 5000,
        'max_cost_device': 10000,
        'min_cost_connection': 150,
        'max_cost_connection': 600,
        'min_capacity_device': 1500,
        'max_capacity_device': 2400,
        'min_data_demand': 200,
        'max_data_demand': 1000,
        'min_n_nodes': 20,
        'max_n_nodes': 30,
        'c_range': (11, 50),
        'd_range': (10, 100),
        'ratio': 100,
        'k_max': 10,
        'er_prob': 0.3,
        'n_network_pairs': 50,
        'network_max_cost': 20,
    }

    network_optimizer = NetworkOptimization(parameters, seed=seed)
    instance = network_optimizer.generate_instance()
    solve_status, solve_time, objective_value = network_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")