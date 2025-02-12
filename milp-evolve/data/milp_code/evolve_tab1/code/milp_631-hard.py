import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EnhancedNetworkOptimization:
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
        penalty_variables = {node: np.random.uniform(self.min_penalty, self.max_penalty) for node in G.nodes}

        maintenance_windows = {node: np.random.choice([0, 1], self.num_time_periods) for node in G.nodes}

        return {
            'G': G,
            'max_packet_effort': max_packet_effort,
            'network_throughput_profit': network_throughput_profit,
            'flow_probabilities': flow_probabilities,
            'critical_machines': critical_machines,
            'compute_efficiencies': compute_efficiencies,
            'machine_restriction': machine_restriction,
            'packet_computation_costs': packet_computation_costs,
            'penalty_variables': penalty_variables,
            'maintenance_windows': maintenance_windows
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
        penalty_variables = instance['penalty_variables']
        maintenance_windows = instance['maintenance_windows']

        model = Model("EnhancedNetworkOptimization")

        packet_effort_vars = {node: model.addVar(vtype="C", name=f"PacketEffort_{node}") for node in G.nodes}
        network_flow_vars = {(u, v): model.addVar(vtype="B", name=f"NetworkFlow_{u}_{v}") for u, v in G.edges}
        critical_machine_flow_vars = {node: model.addVar(vtype="B", name=f"CriticalMachineFlow_{node}") for node in G.nodes}
        restricted_machine_vars = {node: model.addVar(vtype="B", name=f"RestrictedMachine_{node}") for node in G.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"Penalty_{node}") for node in G.nodes}
        high_capacity_utilization_vars = {node: model.addVar(vtype="B", name=f"HighCapacityUtilization_{node}") for node in G.nodes}
        
        maintenance_schedule_vars = {(node, t): model.addVar(vtype="B", name=f"MaintenanceSchedule_{node}_{t}") for node in G.nodes for t in range(self.num_time_periods)}

        critical_flow_vars = {}
        for i, group in enumerate(critical_machines):
            critical_flow_vars[i] = model.addVar(vtype="B", name=f"CriticalFlow_{i}")

        raw_material_allocation_vars = {(node, t): model.addVar(vtype="C", name=f"RawMaterialAllocation_{node}_{t}") for node in G.nodes for t in range(self.num_time_periods)}
        inventory_level_vars = {(node, t): model.addVar(vtype="C", name=f"InventoryLevel_{node}_{t}") for node in G.nodes for t in range(self.num_time_periods)}

        # Objective Function
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
            penalty_vars[node] * penalty_variables[node]
            for node in G.nodes
        )
        for node in G.nodes:
            for t in range(self.num_time_periods):
                objective_expr -= maintenance_schedule_vars[(node, t)] * self.maintenance_penalty

        model.setObjective(objective_expr, "maximize")

        # Constraints
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

        big_M = 1000  # Big M constant; chosen sufficiently large
        for node in G.nodes:
            model.addCons(
                penalty_vars[node] >= packet_effort_vars[node] - big_M * (1 - high_capacity_utilization_vars[node]),
                name=f"BigMPenaltyEffort_{node}"
            )
            model.addCons(
                penalty_vars[node] >= 0,
                name=f"NonNegativePenalty_{node}"
            )
            model.addCons(
                packet_effort_vars[node] <= max_packet_effort[node] + big_M * high_capacity_utilization_vars[node],
                name=f"HighCapacityUtilization_{node}"
            )

        for node in G.nodes:
            for t in range(self.num_time_periods):
                model.addCons(
                    raw_material_allocation_vars[(node, t)] <= self.raw_material_supply[node],
                    name=f"RawMaterialSupply_{node}_{t}"
                )
                if t > 0:
                    model.addCons(
                        inventory_level_vars[(node, t)] == inventory_level_vars[(node, t-1)] + raw_material_allocation_vars[(node, t)] - packet_effort_vars[node],
                        name=f"InventoryBalance_{node}_{t}"
                    )

        for node in G.nodes:
            for t in range(self.num_time_periods):
                if maintenance_windows[node][t]:
                    model.addCons(
                        maintenance_schedule_vars[(node, t)] == 1,
                        name=f"MandatoryMaintenance_{node}_{t}"
                    )
                    
                model.addCons(
                    packet_effort_vars[node] <= (1 - maintenance_schedule_vars[(node, t)]) * big_M,
                    name=f"MaintenanceLimit_{node}_{t}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 70,
        'max_nodes': 562,
        'connection_prob': 0.1,
        'min_flow': 1170,
        'max_flow': 2025,
        'min_compute': 0,
        'max_compute': 2812,
        'min_packet_effort': 1800,
        'max_packet_effort': 2025,
        'min_profit': 750.0,
        'max_profit': 1400.0,
        'max_machine_group_size': 3000,
        'min_efficiency': 150,
        'max_efficiency': 1120,
        'min_computation_cost': 1387,
        'max_computation_cost': 2100,
        'min_penalty': 100,
        'max_penalty': 500,
        'num_time_periods': 10,
        'raw_material_supply': [1500 for _ in range(562)],
        'maintenance_penalty': 2000
    }

    optimizer = EnhancedNetworkOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")