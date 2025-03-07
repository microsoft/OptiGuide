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

        time_windows = {node: (np.random.randint(1, 100), np.random.randint(100, 200)) for node in G.nodes}
        delivery_costs = {(u, v): np.random.uniform(1.0, 50.0) for u, v in G.edges}
        energy_consumption = {(u, v): np.random.uniform(10.0, 100.0) for u, v in G.edges}

        # New Data from Second MILP
        E_invalid = self.generate_incompatibility_data(G)
        routes = self.create_routes(G)
        route_cost = {(u, v): np.random.uniform(5.0, 20.0) for u, v in G.edges}
        feasibility = {(u, v): np.random.randint(0, 2) for u, v in G.edges}
        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        operating_cost = np.random.randint(5, 20, size=n_facilities).tolist()
        capacity = np.random.randint(5, 50, size=n_facilities).tolist()
        patrol_fees = np.random.randint(500, 5000, size=n_facilities).tolist()
        high_priority_nodes = random.sample(list(G.nodes), len(G.nodes) // 3)
        patrol_times = {i: np.random.randint(1, 10, size=n_facilities).tolist() for i in high_priority_nodes}

        # New Data
        demand = {t: np.random.randint(50, 150) for t in range(self.num_periods)}
        incompatibility_matrix = {(u, v): np.random.choice([0, 1], p=[0.7, 0.3]) for u in G.nodes for v in G.nodes if u != v}

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
            'energy_consumption': energy_consumption,
            'E_invalid': E_invalid, 
            'routes': routes,
            'route_cost': route_cost,
            'feasibility': feasibility,
            'n_facilities': n_facilities,
            'operating_cost': operating_cost,
            'capacity': capacity,
            'patrol_fees': patrol_fees,
            'high_priority_nodes': high_priority_nodes,
            'patrol_times': patrol_times,
            'demand': demand,
            'incompatibility_matrix': incompatibility_matrix
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
        E_invalid = instance['E_invalid']
        routes = instance['routes']
        route_cost = instance['route_cost']
        feasibility = instance['feasibility']
        n_facilities = instance['n_facilities']
        operating_cost = instance['operating_cost']
        capacity = instance['capacity']
        patrol_fees = instance['patrol_fees']
        high_priority_nodes = instance['high_priority_nodes']
        patrol_times = instance['patrol_times']
        demand = instance['demand']
        incompatibility_matrix = instance['incompatibility_matrix']

        model = Model("ComplexDistributedNetworkOptimization")

        packet_effort_vars = {node: model.addVar(vtype="C", name=f"PacketEffort_{node}") for node in G.nodes}
        network_flow_vars = {(u, v): model.addVar(vtype="B", name=f"NetworkFlow_{u}_{v}") for u, v in G.edges}
        critical_machine_flow_vars = {node: model.addVar(vtype="B", name=f"CriticalMachineFlow_{node}") for node in G.nodes}
        restricted_machine_vars = {node: model.addVar(vtype="B", name=f"RestrictedMachine_{node}") for node in G.nodes}
        
        critical_flow_vars = {}
        for i, group in enumerate(critical_machines):
            critical_flow_vars[i] = model.addVar(vtype="B", name=f"CriticalFlow_{i}")

        delivery_schedule_vars = {node: model.addVar(vtype="I", name=f"DeliverySchedule_{node}") for node in G.nodes}

        # New variables
        facility_vars = {j: model.addVar(vtype="B", name=f"Facility_{j}") for j in range(n_facilities)}
        assignment_vars = {(node, j): model.addVar(vtype="B", name=f"Assignment_{node}_{j}") for node in G.nodes for j in range(n_facilities)}
        patrol_vars = {(node, j): model.addVar(vtype="B", name=f"Patrol_{node}_{j}") for node in high_priority_nodes for j in range(n_facilities)}

        demand_vars = {t: model.addVar(vtype="I", name=f"Demand_{t}") for t in range(self.num_periods)}
        incompatibility_vars = {(u, v): model.addVar(vtype="B", name=f"Incompatibility_{u}_{v}") for u, v in G.edges}

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
        objective_expr -= quicksum(
            operating_cost[j] * facility_vars[j]
            for j in range(n_facilities)
        )
        objective_expr -= quicksum(
            patrol_fees[j] * patrol_vars[(node, j)]
            for node in high_priority_nodes for j in range(n_facilities)
        )
        objective_expr += total_demand
        objective_expr -= quicksum(
            incompatibility_matrix[(u,v)] * incompatibility_vars[(u,v)]
            for u, v in G.edges
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

        # Demand constraints
        for t in range(self.num_periods):
            model.addCons(
                quicksum(demand[t] * packet_effort_vars[node] for node in G.nodes) >= demand[t],
                name=f"Demand_{t}"
            )
        
        # Incompatibility constraints
        for u, v in G.edges:
            model.addCons(
                incompatibility_vars[(u, v)] <= incompatibility_matrix[(u, v)],
                name=f"Incompatibility_{u}_{v}"
            )
        
        # Capacity constraints
        for j in range(n_facilities):
            model.addCons(
                quicksum(assignment_vars[(node, j)] for node in G.nodes) <= capacity[j] * facility_vars[j],
                name=f"FacilityCapacity_{j}"
            )

        # Patrol constraints for high priority nodes
        for node in high_priority_nodes:
            model.addCons(
                quicksum(patrol_vars[(node, j)] for j in range(n_facilities)) >= 1,
                name=f"PatrolCoverage_{node}"
            )

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
        'zone_prob': 0.1,
        'exclusion_rate': 0.45,
        'campaign_hours': 1680,
        'facility_min_count': 15,
        'facility_max_count': 375,
        'MaxPatrolBudget': 20000,
    }

    optimizer = ComplexDistributedNetworkOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")