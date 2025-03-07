import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class TransportationNetworkOptimization:

    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['traffic'] = np.random.uniform(1, 10)

    def generate_dynamic_conditions(self, G):
        battery_life = {node: np.random.uniform(0.5, 2.0) for node in G.nodes}
        charging_stations = {node: np.random.choice([0, 1], p=[0.3, 0.7]) for node in G.nodes}
        traffic_conditions = {(u, v): np.random.uniform(0.5, 1.5) for u, v in G.edges}
        return battery_life, charging_stations, traffic_conditions

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        battery_life, charging_stations, traffic_conditions = self.generate_dynamic_conditions(G)
        
        hourly_electricity_price = [np.random.uniform(5, 25) for _ in range(24)]
        delivery_deadlines = {node: np.random.randint(1, 24) for node in G.nodes}
        penalty_costs = {node: np.random.uniform(10, 50) for node in G.nodes}
        dynamic_demands = {node: [np.random.randint(1, 10) for _ in range(24)] for node in G.nodes}

        return {
            'G': G,
            'battery_life': battery_life,
            'charging_stations': charging_stations,
            'traffic_conditions': traffic_conditions,
            'hourly_electricity_price': hourly_electricity_price,
            'delivery_deadlines': delivery_deadlines,
            'penalty_costs': penalty_costs,
            'dynamic_demands': dynamic_demands
        }

    def solve(self, instance):
        G = instance['G']
        battery_life = instance['battery_life']
        charging_stations = instance['charging_stations']
        traffic_conditions = instance['traffic_conditions']
        hourly_electricity_price = instance['hourly_electricity_price']
        delivery_deadlines = instance['delivery_deadlines']
        penalty_costs = instance['penalty_costs']
        dynamic_demands = instance['dynamic_demands']
        
        model = Model("TransportationNetworkOptimization")
        node_vars = {f"x{node}": model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        
        drone_energy_vars = {f"e{node}": model.addVar(vtype="C", name=f"e{node}") for node in G.nodes}
        delay_penalty_vars = {f"pen{node}": model.addVar(vtype="C", name=f"pen{node}") for node in G.nodes}
        charging_vars = {(node, t): model.addVar(vtype="B", name=f"charge_{node}_{t}") for node in G.nodes for t in range(24)}

        z_vars = {(node, t): model.addVar(vtype="B", name=f"z{node}_{t}") for node in G.nodes for t in range(24)}
        
        dynamic_demand_vars = {(node, t): model.addVar(vtype="C", name=f"dynamic_demand_{node}_{t}") for node in G.nodes for t in range(24)}

        # Objective Function
        objective_expr = quicksum(
            dynamic_demands[node][t] * node_vars[f"x{node}"]
            for node in G.nodes for t in range(24)
        )

        objective_expr -= quicksum(
            traffic_conditions[(u, v)] * G[u][v]['traffic'] * edge_vars[f"y{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            penalty_costs[node] * delay_penalty_vars[f"pen{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            hourly_electricity_price[t] * charging_vars[(node, t)]
            for node in G.nodes for t in range(24)
        )

        # Constraints
        for u, v in G.edges:
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                name=f"Traffic_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                drone_energy_vars[f"e{node}"] <= battery_life[node],
                name=f"Battery_{node}"
            )

            model.addCons(
                quicksum(charging_vars[(node, t)] for t in range(24)) <= (charging_stations[node] * 24),
                name=f"Charging_{node}"
            )
            
            model.addCons(
                quicksum([charging_vars[(node, t)] for t in range(delivery_deadlines[node])]) >= node_vars[f"x{node}"],
                name=f"Deadline_{node}"
            )
            
            model.addCons(
                drone_energy_vars[f"e{node}"] + delay_penalty_vars[f"pen{node}"] >= delivery_deadlines[node],
                name=f"Penalty_{node}"
            )

            # New constraint: dynamic demand must be at least the static demand
            for t in range(24):
                model.addCons(
                    dynamic_demand_vars[(node, t)] >= dynamic_demands[node][t],
                    name=f"DynamicDemand_{node}_{t}"
                )

        # New constraints involving logical conditions for charging and penalties
        for node in G.nodes:
            for t in range(24):
                model.addCons(
                    z_vars[(node, t)] <= charging_vars[(node, t)],
                    name=f"ChargeActivate_{node}_{t}"
                )
                model.addCons(
                    z_vars[(node, t)] >= (node_vars[f"x{node}"] - 1) * (1 / 24.0),
                    name=f"NodeActive_{node}_{t}"
                )
                model.addCons(
                    z_vars[(node, t)] <= node_vars[f"x{node}"],
                    name=f"DependentCharge_{node}_{t}"
                )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 135,
        'max_n': 675,
        'er_prob': 0.17,
        'battery_limit': 37.5,
        'penalty_limit': 3000,
        'charging_cost_limit': 750,
    }
    # Adjusted new parameters for added variables and constraints
    parameters.update({
        'new_param1': 24,
        'new_param2': 500,
    })
    
    tno = TransportationNetworkOptimization(parameters, seed=seed)
    instance = tno.generate_instance()
    solve_status, solve_time = tno.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")