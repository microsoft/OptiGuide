import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexDistributedNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_network_graph(self):
        n_nodes = np.random.randint(self.min_nodes, self.max_nodes)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.connection_degree, seed=self.seed)
        return G

    def generate_data_flow(self, G):
        for u, v in G.edges:
            G[u][v]['flow'] = np.random.gamma(2.0, 2.0)
        return G

    def generate_machine_data(self, G):
        for node in G.nodes:
            G.nodes[node]['compute_capacity'] = np.random.exponential(scale=1000.0)
        return G

    def get_instance(self):
        G = self.generate_network_graph()
        G = self.generate_data_flow(G)
        G = self.generate_machine_data(G)
        
        max_packet_effort = {node: np.random.lognormal(mean=7, sigma=1.0) for node in G.nodes}
        network_throughput_profit = {node: np.random.beta(2.0, 5.0) * 1000 for node in G.nodes}
        flow_probabilities = {(u, v): np.random.uniform(0.3, 0.9) for u, v in G.edges}
        critical_machines = [set(task) for task in nx.find_cliques(G) if len(task) <= self.max_machine_group_size]
        compute_efficiencies = {node: np.random.poisson(lam=50) for node in G.nodes}
        machine_restriction = {node: np.random.binomial(1, 0.25) for node in G.nodes}

        time_windows = {node: (np.random.randint(1, 100), np.random.randint(100, 200)) for node in G.nodes}
        delivery_costs = {(u, v): np.random.uniform(1.0, 50.0) for u, v in G.edges}
        energy_consumption = {(u, v): np.random.uniform(10.0, 100.0) for u, v in G.edges}

        return {
            'G': G,
            'max_packet_effort': max_packet_effort,
            'network_throughput_profit': network_throughput_profit,
            'flow_probabilities': flow_probabilities,
            'critical_machines': critical_machines,
            'compute_efficiencies': compute_efficiencies,
            'machine_restriction': machine_restriction,
            'time_windows': time_windows,
            'delivery_costs': delivery_costs,
            'energy_consumption': energy_consumption
        }

    def solve(self, instance):
        G = instance['G']
        max_packet_effort = instance['max_packet_effort']
        network_throughput_profit = instance['network_throughput_profit']
        flow_probabilities = instance['flow_probabilities']
        critical_machines = instance['critical_machines']
        compute_efficiencies = instance['compute_efficiencies']
        machine_restriction = instance['machine_restriction']
        time_windows = instance['time_windows']
        delivery_costs = instance['delivery_costs']
        energy_consumption = instance['energy_consumption']

        model = Model("ComplexDistributedNetworkOptimization")

        packet_effort_vars = {node: model.addVar(vtype="C", name=f"PacketEffort_{node}") for node in G.nodes}
        network_flow_vars = {(u, v): model.addVar(vtype="B", name=f"NetworkFlow_{u}_{v}") for u, v in G.edges}
        critical_machine_flow_vars = {node: model.addVar(vtype="B", name=f"CriticalMachineFlow_{node}") for node in G.nodes}
        restricted_machine_vars = {node: model.addVar(vtype="B", name=f"RestrictedMachine_{node}") for node in G.nodes}
        
        critical_flow_vars = {}
        for i, group in enumerate(critical_machines):
            critical_flow_vars[i] = model.addVar(vtype="B", name=f"CriticalFlow_{i}")

        delivery_schedule_vars = {node: model.addVar(vtype="I", name=f"DeliverySchedule_{node}") for node in G.nodes}

        total_demand = quicksum(
            packet_effort_vars[node]
            for node in G.nodes
        )
        
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
        objective_expr -= quicksum(
            delivery_costs[(u, v)] * network_flow_vars[(u, v)]
            for u, v in G.edges
        )
        objective_expr -= quicksum(
            energy_consumption[(u, v)] * network_flow_vars[(u, v)]
            for u, v in G.edges
        )
        objective_expr += total_demand

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
                quicksum(network_flow_vars[(u, v)] for u, v in G.edges if u == node or v == node) <= packet_effort_vars[node],
                name=f"EffortLimit_{node}"
            )

        for node in G.nodes:
            model.addCons(
                time_windows[node][0] <= delivery_schedule_vars[node],
                name=f"TimeWindowStart_{node}"
            )
            model.addCons(
                delivery_schedule_vars[node] <= time_windows[node][1],
                name=f"TimeWindowEnd_{node}"
            )

        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 210,
        'max_nodes': 462,
        'connection_degree': 6,
        'min_flow': 1170,
        'max_flow': 1518,
        'min_compute': 0,
        'max_compute': 2108,
        'max_machine_group_size': 3000,
        'num_periods': 960,
        'vehicle_capacity': 400,
        'regular_working_hours': 1440,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    optimizer = ComplexDistributedNetworkOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")