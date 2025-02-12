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
        capacities = np.random.randint(self.min_station_capacity, self.max_station_capacity + 1, self.n_stations)
        battery_requirements = np.random.gamma(2, 2, self.n_ev_models).astype(int) + 1

        self.time_periods = np.random.randint(1, self.max_time_period + 1)
        energy_costs = np.random.randint(self.min_energy_cost, self.max_energy_cost + 1, (self.time_periods, self.n_stations, self.n_ev_models))
        emissions = np.random.randint(1, 10, self.n_stations)
        tariffs = np.random.uniform(self.min_tariffs, self.max_tariffs, self.n_stations)
        
        graph = Graph.erdos_renyi(self.n_stations, self.zone_probability, seed=self.seed)
        zone_weights = np.random.randint(1, 10, size=len(graph.edges))

        return {
            "installation_costs": installation_costs,
            "energy_costs": energy_costs,
            "capacities": capacities,
            "battery_requirements": battery_requirements,
            "emissions": emissions,
            "tariffs": tariffs,
            "graph": graph,
            "zone_weights": zone_weights
        }

    def solve(self, instance):
        installation_costs = instance['installation_costs']
        energy_costs = instance['energy_costs']
        capacities = instance['capacities']
        battery_requirements = instance['battery_requirements']
        emissions = instance['emissions']
        tariffs = instance['tariffs']
        graph = instance['graph']
        zone_weights = instance['zone_weights']

        model = Model("ElectricVehicleChargingOptimization")
        n_stations = len(installation_costs)
        n_ev_models = len(energy_costs[0][0])
        time_periods = len(energy_costs)

        charging_zone_alloc = {s: model.addVar(vtype="B", name=f"ElectricVehicleCharging_{s}") for s in range(n_stations)}
        ev_model_allocation = {
            (t, s, e): model.addVar(vtype="B", name=f"Station_{s}_EV_{e}_Time_{t}")
            for t in range(time_periods) for s in range(n_stations) for e in range(n_ev_models)
        }

        model.setObjective(
            quicksum(installation_costs[s] * charging_zone_alloc[s] for s in range(n_stations)) +
            quicksum((energy_costs[t][s, e] + tariffs[s]) * ev_model_allocation[(t, s, e)] for t in range(time_periods) for s in range(n_stations) for e in range(n_ev_models)) +
            quicksum(emissions[s] * charging_zone_alloc[s] for s in range(n_stations)),
            "minimize"
        )

        for e in range(n_ev_models):
            model.addCons(quicksum(ev_model_allocation[(t, s, e)] for t in range(time_periods) for s in range(n_stations)) == 1, f"EV_{e}_Allocation")
    
        for s in range(n_stations):
            for e in range(n_ev_models):
                for t in range(time_periods):
                    model.addCons(ev_model_allocation[(t, s, e)] <= charging_zone_alloc[s], f"Station_{s}_Serve_{e}_Time_{t}")
    
        for s in range(n_stations):
            for t in range(time_periods):
                model.addCons(quicksum(battery_requirements[e] * ev_model_allocation[(t, s, e)] for e in range(n_ev_models)) <= capacities[s], f"Station_{s}_Capacity_Time_{t}")

        for edge in graph.edges:
            model.addCons(charging_zone_alloc[edge[0]] + charging_zone_alloc[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for s in range(n_stations):
            model.addCons(
                quicksum(ev_model_allocation[(t, s, e)] for t in range(time_periods) for e in range(n_ev_models)) <= time_periods * n_ev_models * charging_zone_alloc[s],
                f"Convex_Hull_{s}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_stations': 18,
        'n_ev_models': 378,
        'min_energy_cost': 464,
        'max_energy_cost': 3000,
        'min_installation_cost': 1047,
        'max_installation_cost': 5000,
        'min_station_capacity': 59,
        'max_station_capacity': 2154,
        'zone_probability': 0.31,
        'max_time_period': 5,
        'min_tariffs': 0.59,
        'max_tariffs': 0.59,
    }

    resource_optimizer = ElectricVehicleChargingOptimization(parameters, seed=seed)
    instance = resource_optimizer.get_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")