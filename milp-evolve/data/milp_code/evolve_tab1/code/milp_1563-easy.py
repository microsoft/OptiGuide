import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyRouteOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        if self.graph_type == 'ER':
            G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        elif self.graph_type == 'BA':
            G = nx.barabasi_albert_graph(n=n_nodes, m=self.barabasi_m, seed=self.seed)
        return G

    def generate_emergency_costs_times(self, G):
        for node in G.nodes:
            G.nodes[node]['incident_severity'] = np.random.randint(1, 100)
            G.nodes[node]['emergency_cost'] = np.random.randint(1, 50)

        for u, v in G.edges:
            G[u][v]['travel_time'] = (G.nodes[u]['incident_severity'] + G.nodes[v]['incident_severity']) / float(self.time_param)

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_emergency_costs_times(G)
        res = {'G': G}
        
        for u, v in G.edges:
            travel_times = np.linspace(0, G[u][v]['travel_time'], self.num_segments+1)
            segments = np.diff(travel_times)
            res[f'travel_times_{u}_{v}'] = travel_times
            res[f'segments_{u}_{v}'] = segments
        
        for node in G.nodes:
            res[f'fixed_cost_{node}'] = np.random.randint(100, 200)
            res[f'variable_cost_{node}'] = np.random.uniform(1, 5)
            res[f'zone_{node}'] = np.random.choice(['A', 'B', 'C']) 
            res[f'priority_{node}'] = np.random.uniform(1, 10) # priority for emergency response

        for u, v in G.edges:
            res[f'traffic_factor_{u}_{v}'] = np.random.uniform(0.5, 5) # traffic factor for route

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['G']
        
        model = Model("EmergencyRouteOptimization")
        vehicle_vars = {f"v{node}":  model.addVar(vtype="B", name=f"v{node}") for node in G.nodes}
        route_vars = {f"r{u}_{v}": model.addVar(vtype="B", name=f"r{u}_{v}") for u, v in G.edges}
        segment_vars = {f"s{u}_{v}_{k}": model.addVar(vtype="C", lb=0, name=f"s{u}_{v}_{k}") for u, v in G.edges for k in range(self.num_segments)}
        buffer_vars = {f"b{node}": model.addVar(vtype="B", name=f"b{node}") for node in G.nodes} # buffer time variables
        zone_vars = {f"z{node}": model.addVar(vtype="B", name=f"z{node}") for node in G.nodes} # zone compliance variables

        # Modified objective function
        objective_expr = quicksum(
            G.nodes[node]['incident_severity'] * vehicle_vars[f"v{node}"]
            for node in G.nodes
        ) - quicksum(
            instance[f'segments_{u}_{v}'][k] * segment_vars[f"s{u}_{v}_{k}"]
            for u, v in G.edges
            for k in range(self.num_segments)
        )

        # Applying Piecewise Linear Function Constraints for travel time
        M = 10000  # Big M constant, should be large enough
        for u, v in G.edges:
            model.addCons(
                vehicle_vars[f"v{u}"] + vehicle_vars[f"v{v}"] - route_vars[f"r{u}_{v}"] * M <= 1,
                name=f"R_{u}_{v}"
            )
                
        for u, v in G.edges:
            model.addCons(quicksum(segment_vars[f"s{u}_{v}_{k}"] for k in range(self.num_segments)) == route_vars[f'r{u}_{v}'])
            
            for k in range(self.num_segments):
                model.addCons(segment_vars[f"s{u}_{v}_{k}"] <= instance[f'travel_times_{u}_{v}'][k+1] - instance[f'travel_times_{u}_{v}'][k] + (1 - route_vars[f'r{u}_{v}']) * M)

        # Buffer time constraints
        for node in G.nodes:
            model.addCons(
                vehicle_vars[f"v{node}"] * instance[f'fixed_cost_{node}'] + vehicle_vars[f"v{node}"] * instance[f'variable_cost_{node}'] <= self.emergency_budget
            )

        model.addCons(
            quicksum(instance[f'priority_{node}'] * buffer_vars[f"b{node}"] for node in G.nodes) <= self.response_capacity,
            name="BufferTime"
        )

        model.addCons(
            quicksum(instance[f'traffic_factor_{u}_{v}'] * route_vars[f"r{u}_{v}"] for u, v in G.edges) <= self.response_capacity,
            name="ZoneRestrictions"
        )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 81,
        'max_n': 1536,
        'er_prob': 0.8,
        'graph_type': 'BA',
        'barabasi_m': 5,
        'time_param': 3000.0,
        'num_segments': 30,
        'emergency_budget': 50000,
        'response_capacity': 750,
    }

    ero = EmergencyRouteOptimization(parameters, seed=seed)
    instance = ero.generate_instance()
    solve_status, solve_time = ero.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")