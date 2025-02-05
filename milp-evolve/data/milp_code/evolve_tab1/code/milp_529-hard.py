import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class DistributedNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_network_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.connection_prob, directed=True, seed=self.seed)
        return G

    def generate_data_flow(self, G):
        for u, v in G.edges:
            G[u][v]['flow'] = np.random.uniform(self.min_flow, self.max_flow)
        return G

    def generate_machine_data(self, G):
        for node in G.nodes:
            G.nodes[node]['compute_capacity'] = np.random.randint(self.min_compute, self.max_compute)
        return G

    def get_instance(self):
        G = self.generate_network_graph()
        G = self.generate_data_flow(G)
        G = self.generate_machine_data(G)
        
        max_packet_effort = {node: np.random.randint(self.min_packet_effort, self.max_packet_effort) for node in G.nodes}
        network_throughput_profit = {node: np.random.uniform(self.min_profit, self.max_profit) for node in G.nodes}
        flow_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        critical_machines = [set(task) for task in nx.find_cliques(G.to_undirected()) if len(task) <= self.max_machine_group_size]
        compute_efficiencies = {node: np.random.randint(self.min_efficiency, self.max_efficiency) for node in G.nodes}
        machine_restriction = {node: np.random.randint(0, 2) for node in G.nodes}
        
        packet_computation_costs = {node: np.random.randint(self.min_computation_cost, self.max_computation_cost) for node in G.nodes}
        machine_capacity_limits = {node: np.random.randint(self.min_capacity_limit, self.max_capacity_limit) for node in G.nodes}
        
        extra_flow_probabilities = {(u, v): np.random.uniform(0.1, 0.5) for u, v in G.edges}
        
        return {
            'G': G,
            'max_packet_effort': max_packet_effort,
            'network_throughput_profit': network_throughput_profit,
            'flow_probabilities': flow_probabilities,
            'critical_machines': critical_machines,
            'compute_efficiencies': compute_efficiencies,
            'machine_restriction': machine_restriction,
            'packet_computation_costs': packet_computation_costs,
            'machine_capacity_limits': machine_capacity_limits,
            'extra_flow_probabilities': extra_flow_probabilities
        }

    def solve(self, instance):
        G = instance['G']
        max_packet_effort = instance['max_packet_effort']
        network_throughput_profit = instance['network_throughput_profit']
        flow_probabilities = instance['flow_probabilities']
        critical_machines = instance['critical_machines']
        compute_efficiencies = instance['compute_efficiencies']
        machine_restriction = instance['machine_restriction']
        packet_computation_costs = instance['packet_computation_costs']
        machine_capacity_limits = instance['machine_capacity_limits']
        extra_flow_probabilities = instance['extra_flow_probabilities']

        model = Model("DistributedNetworkOptimization")

        packet_effort_vars = {node: model.addVar(vtype="C", name=f"PacketEffort_{node}") for node in G.nodes}
        network_flow_vars = {(u, v): model.addVar(vtype="B", name=f"NetworkFlow_{u}_{v}") for u, v in G.edges}
        critical_machine_flow_vars = {node: model.addVar(vtype="B", name=f"CriticalMachineFlow_{node}") for node in G.nodes}
        restricted_machine_vars = {node: model.addVar(vtype="B", name=f"RestrictedMachine_{node}") for node in G.nodes}

        extra_flow_vars = {(u, v): model.addVar(vtype="B", name=f"ExtraFlow_{u}_{v}") for u, v in G.edges}

        critical_flow_vars = {}
        for i, group in enumerate(critical_machines):
            critical_flow_vars[i] = model.addVar(vtype="B", name=f"CriticalFlow_{i}")

        objective_expr = quicksum(
            network_throughput_profit[node] * packet_effort_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['flow'] * network_flow_vars[(u, v)]
            for u, v in G.edges
        )
        objective_expr += quicksum(
            compute_efficiencies[node] * critical_machine_flow_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            machine_restriction[node] * restricted_machine_vars[node]
            for node in G.nodes
        )

        for node in G.nodes:
            model.addCons(
                packet_effort_vars[node] <= max_packet_effort[node],
                name=f"MaxPacketEffort_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                network_flow_vars[(u, v)] <= flow_probabilities[(u, v)],
                name=f"FlowProbability_{u}_{v}"
            )
            model.addCons(
                network_flow_vars[(u, v)] <= packet_effort_vars[u],
                name=f"FlowAssignLimit_{u}_{v}"
            )

        for group in critical_machines:
            model.addCons(
                quicksum(critical_machine_flow_vars[node] for node in group) <= 1,
                name=f"MaxOneCriticalFlow_{group}"
            )

        for node in G.nodes:
            model.addCons(
                restricted_machine_vars[node] <= machine_restriction[node],
                name=f"MachineRestriction_{node}"
            )
            
        for node in G.nodes:
            model.addCons(
                quicksum(network_flow_vars[(u, v)] for u, v in G.edges if u == node or v == node) <= machine_capacity_limits[node],
                name=f"CapacityLimit_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                extra_flow_vars[(u, v)] <= extra_flow_probabilities[(u, v)],
                name=f"ExtraFlowProbability_{u}_{v}"
            )
            
        for u, v in G.edges:
            model.addCons(
                network_flow_vars[(u, v)] + extra_flow_vars[(u, v)] <= 1,
                name=f"SetPacking_{u}_{v}"
            )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 35,
        'max_nodes': 562,
        'connection_prob': 0.17,
        'min_flow': 390,
        'max_flow': 2025,
        'min_compute': 0,
        'max_compute': 1406,
        'min_packet_effort': 1800,
        'max_packet_effort': 2025,
        'min_profit': 375.0,
        'max_profit': 2800.0,
        'max_machine_group_size': 3000,
        'min_efficiency': 75,
        'max_efficiency': 112,
        'min_computation_cost': 1850,
        'max_computation_cost': 2800,
        'min_capacity_limit': 200,
        'max_capacity_limit': 1125,
    }
    
    optimizer = DistributedNetworkOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")