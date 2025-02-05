import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class NetworkFlowOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.min_demand >= 0 and self.max_demand >= self.min_demand
        assert self.min_capacity >= 0 and self.max_capacity >= self.min_capacity

        # Generate random node capacities
        node_capacities = self.min_capacity + (self.max_capacity - self.min_capacity) * np.random.rand(self.n_nodes)
        # Generate random demands for each data packet
        data_demands = self.min_demand + (self.max_demand - self.min_demand) * np.random.rand(self.n_packets)
        # Generate random transmission costs for each link
        transmission_costs = np.random.rand(self.n_packets, self.n_nodes)
        # Generate random node maintenance costs
        node_costs = np.random.exponential(50, size=self.n_nodes).tolist()

        # Generate random demand clusters
        demand_clusters = np.random.randint(1, 5, self.n_packets)
        # Generate random node sets for each cluster
        node_sets = {i: np.random.choice(self.n_nodes, int(self.n_nodes/3), replace=False) for i in range(1, 6)}

        return {
            "node_capacities": node_capacities,
            "data_demands": data_demands,
            "transmission_costs": transmission_costs,
            "node_costs": node_costs,
            "demand_clusters": demand_clusters,
            "node_sets": node_sets
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        node_capacities = instance['node_capacities']
        data_demands = instance['data_demands']
        transmission_costs = instance['transmission_costs']
        node_costs = instance['node_costs']
        demand_clusters = instance['demand_clusters']
        node_sets = instance['node_sets']

        model = Model("NetworkFlowOptimization")

        flow_vars = {(i, j): model.addVar(vtype="C", name=f"flow_{i}_{j}") for i in range(self.n_nodes) for j in range(self.n_packets)}
        node_active_vars = {i: model.addVar(vtype="B", name=f"active_{i}") for i in range(self.n_nodes)}
        set_active_vars = {i: model.addVar(vtype="B", name=f"set_active_{i}") for i in node_sets.keys()}

        # Objective: minimize the total cost (node maintenance + transmission)
        objective_expr = (
            quicksum(node_costs[i] * node_active_vars[i] for i in range(self.n_nodes)) +
            quicksum(transmission_costs[j][i] * flow_vars[i, j] for i in range(self.n_nodes) for j in range(self.n_packets)) +
            quicksum(set_active_vars[i] for i in node_sets.keys()) * 10  # Arbitrary cost for activating a set
        )

        # Add constraints
        # Constraint: All data packet demands must be satisfied
        for j in range(self.n_packets):
            model.addCons(quicksum(flow_vars[i, j] for i in range(self.n_nodes)) == data_demands[j], f"Demand_{j}")

        # Constraint: Node capacity must not be exceeded
        for i in range(self.n_nodes):
            model.addCons(quicksum(flow_vars[i, j] for j in range(self.n_packets)) <= node_capacities[i] * node_active_vars[i], f"Capacity_{i}")

        # Constraint: Linking node activation to flow
        for i in range(self.n_nodes):
            model.addCons(node_active_vars[i] <= 1, f"NodeActive_{i}")
 
        # Set Packing Constraints: Ensure nodes are efficiently used
        for cluster_id, nodes in node_sets.items():
            model.addCons(quicksum(node_active_vars[i] for i in nodes) >= set_active_vars[cluster_id], f"SetActive_{cluster_id}")
            for j in range(self.n_packets):
                if demand_clusters[j] == cluster_id:
                    model.addCons(quicksum(flow_vars[i, j] for i in nodes) <= set_active_vars[cluster_id] * data_demands[j], f"SetFlow_{cluster_id}_{j}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 150,
        'n_packets': 100,
        'min_demand': 800,
        'max_demand': 2000,
        'min_capacity': 2025,
        'max_capacity': 3000,
        'num_demand_clusters': 75,
        'num_nodes_per_cluster': 'int(200 / 3)',
        'set_activation_cost': 10,
    }
    
    network_flow_optimization = NetworkFlowOptimization(parameters, seed=seed)
    instance = network_flow_optimization.generate_instance()
    solve_status, solve_time = network_flow_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")