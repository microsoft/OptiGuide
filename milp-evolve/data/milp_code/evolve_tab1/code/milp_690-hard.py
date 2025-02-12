import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedDistributedNetworkOptimization:
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
            G.nodes[node]['campaign_budget'] = np.random.randint(100, 500)
        return G

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_routes(self, G):
        routes = list(nx.find_cliques(G.to_undirected()))
        return routes

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

        energy_consumption = {(u, v): np.random.uniform(10.0, 100.0) for u, v in G.edges}

        E_invalid = self.generate_incompatibility_data(G)
        routes = self.create_routes(G)
        route_cost = {(u, v): np.random.uniform(5.0, 20.0) for u, v in G.edges}
        feasibility = {(u, v): np.random.randint(0, 2) for u, v in G.edges}
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.randint(5, 20, size=n_facilities).tolist()
        capacity = np.random.randint(5, 50, size=n_facilities).tolist()

        return {
            'G': G,
            'max_packet_effort': max_packet_effort,
            'network_throughput_profit': network_throughput_profit,
            'flow_probabilities': flow_probabilities,
            'critical_machines': critical_machines,
            'compute_efficiencies': compute_efficiencies,
            'machine_restriction': machine_restriction,
            'energy_consumption': energy_consumption,
            'E_invalid': E_invalid,
            'routes': routes,
            'route_cost': route_cost,
            'feasibility': feasibility,
            'n_facilities': n_facilities,
            'operating_cost': operating_cost,
            'capacity': capacity,
        }

    def solve(self, instance):
        G = instance['G']
        max_packet_effort = instance['max_packet_effort']
        network_throughput_profit = instance['network_throughput_profit']
        flow_probabilities = instance['flow_probabilities']
        critical_machines = instance['critical_machines']
        compute_efficiencies = instance['compute_efficiencies']
        machine_restriction = instance['machine_restriction']
        energy_consumption = instance['energy_consumption']
        E_invalid = instance['E_invalid']
        routes = instance['routes']
        route_cost = instance['route_cost']
        feasibility = instance['feasibility']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        capacity = instance['capacity']

        model = Model("SimplifiedDistributedNetworkOptimization")

        packet_effort_vars = {node: model.addVar(vtype="C", name=f"PacketEffort_{node}") for node in G.nodes}
        network_flow_vars = {(u, v): model.addVar(vtype="B", name=f"NetworkFlow_{u}_{v}") for u, v in G.edges}
        critical_machine_flow_vars = {node: model.addVar(vtype="B", name=f"CriticalMachineFlow_{node}") for node in G.nodes}
        restricted_machine_vars = {node: model.addVar(vtype="B", name=f"RestrictedMachine_{node}") for node in G.nodes}
        
        critical_flow_vars = {}
        for i, group in enumerate(critical_machines):
            critical_flow_vars[i] = model.addVar(vtype="B", name=f"CriticalFlow_{i}")

        # New variables
        facility_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        assignment_vars = {(node, j): model.addVar(vtype="B", name=f"Assignment_{node}_{j}") for node in G.nodes for j in range(n_facilities)}

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
            operating_cost[j] * facility_vars[j]
            for j in range(n_facilities)
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
        
        # Capacity constraints
        for j in range(n_facilities):
            model.addCons(
                quicksum(assignment_vars[(node, j)] for node in G.nodes) <= capacity[j] * facility_vars[j],
                name=f"FacilityCapacity_{j}"
            )

        ### Removed patrol constraints and variables ###
        ### Removed fair labor constraints and variables ###
        ### Removed geographical diversity constraints ###
        ### Removed delivery schedule constraints and variables ###

        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_nodes': 52,
        'max_nodes': 1155,
        'connection_degree': 4,
        'min_flow': 585,
        'max_flow': 1518,
        'min_compute': 0,
        'max_compute': 1581,
        'max_machine_group_size': 3000,
        'num_periods': 960,
        'vehicle_capacity': 600,
        'regular_working_hours': 540,
        'zone_prob': 0.59,
        'exclusion_rate': 0.73,
        'campaign_hours': 1260,
        'facility_min_count': 105,
        'facility_max_count': 1968,
    }
    optimizer = SimplifiedDistributedNetworkOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")