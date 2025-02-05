import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, probability, seed=None):
        graph = nx.erdos_renyi_graph(number_of_nodes, probability, seed=seed)
        edges = set(graph.edges)
        degrees = [d for (n, d) in graph.degree]
        neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes}
        return Graph(number_of_nodes, edges, degrees, neighbors)

class ElectricVehicleChargingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def get_instance(self):
        installation_costs = np.random.randint(self.min_installation_cost, self.max_installation_cost + 1, self.n_stations)
        energy_costs = np.random.randint(self.min_energy_cost, self.max_energy_cost + 1, (self.n_stations, self.n_ev_models))
        capacities = np.random.randint(self.min_station_capacity, self.max_station_capacity + 1, self.n_stations)
        battery_requirements = np.random.gamma(2, 2, self.n_ev_models).astype(int) + 1

        # Introduce piecewise capacity costs, say 3 segments
        pw_linear_segments = []
        for _ in range(self.n_stations):
            segments = np.random.randint(self.min_segment_cost, self.max_segment_cost + 1, self.n_segments)
            pw_linear_segments.append(sorted(segments))
        
        graph = Graph.erdos_renyi(self.n_stations, self.zone_probability, seed=self.seed)
        zone_weights = np.random.randint(1, 10, size=len(graph.edges))

        return {
            "installation_costs": installation_costs,
            "energy_costs": energy_costs,
            "capacities": capacities,
            "battery_requirements": battery_requirements,
            "pw_linear_segments": pw_linear_segments,
            "graph": graph,
            "zone_weights": zone_weights
        }

    def solve(self, instance):
        installation_costs = instance['installation_costs']
        energy_costs = instance['energy_costs']
        capacities = instance['capacities']
        battery_requirements = instance['battery_requirements']
        pw_linear_segments = instance['pw_linear_segments']
        graph = instance['graph']
        zone_weights = instance['zone_weights']

        model = Model("ElectricVehicleChargingOptimization")
        n_stations = len(installation_costs)
        n_ev_models = len(energy_costs[0])
        n_segments = len(pw_linear_segments[0])

        charging_zone_alloc = {s: model.addVar(vtype="B", name=f"ElectricVehicleCharging_{s}") for s in range(n_stations)}
        ev_model_allocation = {(s, e): model.addVar(vtype="B", name=f"Station_{s}_EV_{e}") for s in range(n_stations) for e in range(n_ev_models)}

        seg_costs = {(s, k): model.addVar(vtype="C", name=f"SegCost_{s}_{k}") for s in range(n_stations) for k in range(n_segments)}
        seg_usage = {(s, k): model.addVar(vtype="B", name=f"SegUsage_{s}_{k}") for s in range(n_stations) for k in range(n_segments)}

        model.setObjective(
            quicksum(installation_costs[s] * charging_zone_alloc[s] for s in range(n_stations)) +
            quicksum(energy_costs[s, e] * ev_model_allocation[s, e] for s in range(n_stations) for e in range(n_ev_models)) +
            quicksum(seg_costs[s, k] for s in range(n_stations) for k in range(n_segments)),
            "minimize"
        )

        for e in range(n_ev_models):
            model.addCons(quicksum(ev_model_allocation[s, e] for s in range(n_stations)) == 1, f"EV_{e}_Allocation")
    
        for s in range(n_stations):
            for e in range(n_ev_models):
                model.addCons(ev_model_allocation[s, e] <= charging_zone_alloc[s], f"Station_{s}_Serve_{e}")
    
        for s in range(n_stations):
            model.addCons(quicksum(seg_costs[s, k] for k in range(n_segments)) <= capacities[s], f"Station_{s}_Segmented_Capacity")

        for edge in graph.edges:
            model.addCons(charging_zone_alloc[edge[0]] + charging_zone_alloc[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for s in range(n_stations):
            model.addCons(
                quicksum(ev_model_allocation[s, e] for e in range(n_ev_models)) <= n_ev_models * charging_zone_alloc[s],
                f"Convex_Hull_{s}"
            )

            for k in range(n_segments):
                model.addCons(
                    seg_costs[s, k] == pw_linear_segments[s][k] * seg_usage[s, k],
                    f"SegmentCost_{s}_{k}"
                )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_stations': 24,
        'n_ev_models': 252,
        'min_energy_cost': 1856,
        'max_energy_cost': 3000,
        'min_installation_cost': 1570,
        'max_installation_cost': 5000,
        'min_station_capacity': 590,
        'max_station_capacity': 1615,
        'zone_probability': 0.52,
        'min_segment_cost': 100,
        'max_segment_cost': 2500,
        'n_segments': 2,
    }

    resource_optimizer = ElectricVehicleChargingOptimization(parameters, seed=seed)
    instance = resource_optimizer.get_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")