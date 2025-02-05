import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ModelDeliveryOptimization:
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
        
        perishability_vars = {node: np.random.randint(0, 2) for node in G.nodes}
        seasonal_demand_fluctuations = {t: np.random.normal(loc=1.0, scale=0.1) for t in range(self.num_periods)}
        weather_based_storage_costs = {node: np.random.uniform(self.min_storage_cost, self.max_storage_cost) for node in G.nodes}

        # New Data for Enhanced Complexity
        drone_prob = 0.4
        drone_compatible_edges = {(u, v): np.random.choice([0, 1], p=[1-drone_prob, drone_prob]) for u, v in G.edges}
        energy_consumption = {(u, v): np.random.uniform(5, 20) for u, v in G.edges}
        zone_storage = {node: np.random.randint(self.min_storage, self.max_storage) for node in G.nodes}
        hybrid_flow_savings = {(u, v): np.random.uniform(1, 5) for u, v in G.edges}

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
            'extra_flow_probabilities': extra_flow_probabilities,
            'perishability_vars': perishability_vars,
            'seasonal_demand_fluctuations': seasonal_demand_fluctuations,
            'weather_based_storage_costs': weather_based_storage_costs,
            'drone_compatible_edges': drone_compatible_edges,
            'energy_consumption': energy_consumption,
            'zone_storage': zone_storage,
            'hybrid_flow_savings': hybrid_flow_savings
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
        perishability_vars = instance['perishability_vars']
        seasonal_demand_fluctuations = instance['seasonal_demand_fluctuations']
        weather_based_storage_costs = instance['weather_based_storage_costs']
        drone_compatible_edges = instance['drone_compatible_edges']
        energy_consumption = instance['energy_consumption']
        zone_storage = instance['zone_storage']
        hybrid_flow_savings = instance['hybrid_flow_savings']

        model = Model("ModelDeliveryOptimization")

        packet_effort_vars = {node: model.addVar(vtype="C", name=f"PacketEffort_{node}") for node in G.nodes}
        network_flow_vars = {(u, v): model.addVar(vtype="B", name=f"NetworkFlow_{u}_{v}") for u, v in G.edges}
        critical_machine_flow_vars = {node: model.addVar(vtype="B", name=f"CriticalMachineFlow_{node}") for node in G.nodes}
        restricted_machine_vars = {node: model.addVar(vtype="B", name=f"RestrictedMachine_{node}") for node in G.nodes}
        extra_flow_vars = {(u, v): model.addVar(vtype="B", name=f"ExtraFlow_{u}_{v}") for u, v in G.edges}
        
        critical_flow_vars = {}
        for i, group in enumerate(critical_machines):
            critical_flow_vars[i] = model.addVar(vtype="B", name=f"CriticalFlow_{i}")

        perishability_flow_vars = {node: model.addVar(vtype="B", name=f"PerishabilityFlow_{node}") for node in G.nodes}
        drone_delivery_vars = {(u, v): model.addVar(vtype="B", name=f"DroneDelivery_{u}_{v}") for u, v in G.edges}
        hybrid_flow_vars = {(u, v): model.addVar(vtype="B", name=f"HybridFlow_{u}_{v}") for u, v in G.edges}

        total_demand = quicksum(
            seasonal_demand_fluctuations[time_period] * packet_effort_vars[node]
            for node in G.nodes
            for time_period in range(self.num_periods)
        )
        
        total_storage_costs = quicksum(
            weather_based_storage_costs[node] * perishability_flow_vars[node]
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
        objective_expr += total_demand
        objective_expr -= total_storage_costs

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

        for node in G.nodes:
            model.addCons(
                perishability_flow_vars[node] <= perishability_vars[node],
                name=f"PerishabilityConstraint_{node}"
            )

        # New Constraints
        for u, v in G.edges:
            model.addCons(
                drone_delivery_vars[(u, v)] <= drone_compatible_edges[(u, v)],
                name=f"DroneDeliveryConstraint_{u}_{v}"
            )

        # Prioritization of drone delivery routes
        total_routes = quicksum(network_flow_vars[(u, v)] + extra_flow_vars[(u, v)] for u, v in G.edges)
        drone_routes = quicksum(drone_delivery_vars[(u, v)] for u, v in G.edges)
        model.addCons(drone_routes >= 0.3 * total_routes, name="MinDroneRoutes")

        # Zonal storage constraints
        for node in G.nodes:
            model.addCons(
                quicksum(packet_effort_vars[node] for node in G.nodes if node in G.neighbors(node)) <= zone_storage[node],
                name=f"ZonalStorageCapacity_{node}"
            )

        # Hybrid flow constraints
        for u, v in G.edges:
            model.addCons(
                hybrid_flow_vars[(u, v)] <= hybrid_flow_savings[(u, v)],
                name=f"HybridFlowSavings_{u}_{v}"
            )

        # Costly flow constraints
        costly_prob = 0.2
        for u, v in G.edges:
            model.addCons(
                network_flow_vars[(u, v)] + extra_flow_vars[(u, v)] + hybrid_flow_vars[(u, v)] <= costly_prob,
                name=f"CostlyFlows_{u}_{v}"
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
        'min_nodes': 26,
        'max_nodes': 281,
        'connection_prob': 0.24,
        'min_flow': 585,
        'max_flow': 759,
        'min_compute': 0,
        'max_compute': 527,
        'min_packet_effort': 675,
        'max_packet_effort': 1138,
        'min_profit': 281.25,
        'max_profit': 1400.0,
        'max_machine_group_size': 3000,
        'min_efficiency': 18,
        'max_efficiency': 588,
        'min_computation_cost': 693,
        'max_computation_cost': 1050,
        'min_capacity_limit': 750,
        'max_capacity_limit': 2529,
        'num_periods': 120,
        'min_storage_cost': 225,
        'max_storage_cost': 3000,
        'min_storage': 300,
        'max_storage': 900,
        'vehicle_capacity': 100,
        'regular_working_hours': 720,
    }
    ### given parameter code ends here
    ### new parameter code ends here

    optimizer = ModelDeliveryOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")