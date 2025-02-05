import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EVChargingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.num_zones > 0 and self.num_stations > 0
        assert self.min_station_cost >= 0 and self.max_station_cost >= self.min_station_cost
        assert self.min_charging_cost >= 0 and self.max_charging_cost >= self.min_charging_cost
        assert self.min_energy_capacity > 0 and self.max_energy_capacity >= self.min_energy_capacity
        assert self.max_distance >= 0

        station_costs = np.random.randint(self.min_station_cost, self.max_station_cost + 1, self.num_stations)
        charging_costs = np.random.uniform(self.min_charging_cost, self.max_charging_cost, (self.num_stations, self.num_zones))
        energy_capacities = np.random.randint(self.min_energy_capacity, self.max_energy_capacity + 1, self.num_stations)
        zone_demand = np.random.randint(1, 10, self.num_zones)
        distances = np.random.uniform(0, self.max_distance, (self.num_stations, self.num_zones))

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.num_stations):
            for d in range(self.num_zones):
                G.add_edge(f"station_{p}", f"zone_{d}")
                node_pairs.append((f"station_{p}", f"zone_{d}"))

        equipment_varieties = 2  # Reduced number of equipment varieties
        variety_zone_demand = np.random.randint(1, 10, (equipment_varieties, self.num_zones))

        equipment_costs = np.random.randint(self.min_equipment_cost, self.max_equipment_cost + 1, self.num_stations)
        link_establishment_costs = np.random.randint(self.min_link_cost, self.max_link_cost + 1, (self.num_stations, self.num_zones))
        buffer_stocks = np.random.randint(1, 50, (self.num_stations, self.num_zones))

        return {
            "station_costs": station_costs,
            "charging_costs": charging_costs,
            "energy_capacities": energy_capacities,
            "zone_demand": zone_demand,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "equipment_varieties": equipment_varieties,
            "variety_zone_demand": variety_zone_demand,
            "equipment_costs": equipment_costs,
            "link_establishment_costs": link_establishment_costs,
            "buffer_stocks": buffer_stocks,
        }

    def solve(self, instance):
        station_costs = instance['station_costs']
        charging_costs = instance['charging_costs']
        energy_capacities = instance['energy_capacities']
        zone_demand = instance['zone_demand']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        equipment_varieties = instance['equipment_varieties']
        variety_zone_demand = instance['variety_zone_demand']
        equipment_costs = instance['equipment_costs']
        link_establishment_costs = instance['link_establishment_costs']
        buffer_stocks = instance['buffer_stocks']

        model = Model("EVChargingOptimization")
        num_stations = len(station_costs)
        num_zones = len(charging_costs[0])

        # Decision variables
        station_activation_vars = {p: model.addVar(vtype="B", name=f"Station_{p}") for p in range(num_stations)}
        charging_vars = {(u, v): model.addVar(vtype="I", name=f"Charging_{u}_{v}") for u, v in node_pairs}
        variety_station_vars = {(vtype, u, v): model.addVar(vtype="C", name=f"Equipment_{vtype}_{u}_{v}") for vtype in range(equipment_varieties) for u, v in node_pairs}
        equipment_activation_vars = {p: model.addVar(vtype="B", name=f"Equipment_{p}") for p in range(num_stations)}
        link_activation_vars = {(p, d): model.addVar(vtype="B", name=f"Link_{p}_{d}") for p in range(num_stations) for d in range(num_zones)}
        buffer_vars = {(p, d): model.addVar(vtype="I", name=f"Buffer_{p}_{d}") for p in range(num_stations) for d in range(num_zones)}

        model.setObjective(
            quicksum(station_costs[p] * station_activation_vars[p] for p in range(num_stations)) +
            quicksum(charging_costs[int(u.split('_')[1]), int(v.split('_')[1])] * charging_vars[(u, v)] for (u, v) in node_pairs) +
            quicksum(equipment_costs[p] * equipment_activation_vars[p] for p in range(num_stations)) +
            quicksum(link_establishment_costs[p, d] * link_activation_vars[p, d] for p in range(num_stations) for d in range(num_zones)) +
            quicksum(buffer_vars[(p, d)] / buffer_stocks[p, d] for p in range(num_stations) for d in range(num_zones)), 
            "minimize"
        )

        # EV charging equipment distribution constraint for each variety at each zone
        for d in range(num_zones):
            for vtype in range(equipment_varieties):
                model.addCons(
                    quicksum(variety_station_vars[(vtype, u, f"zone_{d}")] for u in G.predecessors(f"zone_{d}")) >= variety_zone_demand[vtype, d], 
                    f"Zone_{d}_NodeFlowConservation_{vtype}"
                )

        # Constraints: Stations cannot exceed their energy capacities
        for p in range(num_stations):
            model.addCons(
                quicksum(variety_station_vars[(vtype, f"station_{p}", f"zone_{d}")] for d in range(num_zones) for vtype in range(equipment_varieties)) <= energy_capacities[p], 
                f"Station_{p}_MaxEnergyCapacity"
            )

        # Coverage constraint for all zones based on Euclidean distances
        for d in range(num_zones):
            model.addCons(
                quicksum(station_activation_vars[p] for p in range(num_stations) if distances[p, d] <= self.max_distance) >= 1, 
                f"Zone_{d}_Coverage"
            )

        # Constraints: Charging cannot exceed its capacity
        for u, v in node_pairs:
            model.addCons(charging_vars[(u, v)] <= buffer_stocks[int(u.split('_')[1]), int(v.split('_')[1])], f"ChargingCapacity_{u}_{v}")

        # Network design constraints: Each zone is served by exactly one charging station
        for d in range(num_zones):
            model.addCons(
                quicksum(link_activation_vars[p, d] for p in range(num_stations)) == 1, 
                f"Zone_{d}_StationAssignment"
            )

        # Constraints: Only active stations can have active links
        for p in range(num_stations):
            for d in range(num_zones):
                model.addCons(
                    link_activation_vars[p, d] <= equipment_activation_vars[p], 
                    f"Station_{p}_Link_{d}"
                )

        # Equipment capacity constraints
        for p in range(num_stations):
            model.addCons(
                quicksum(zone_demand[d] * link_activation_vars[p, d] for d in range(num_zones)) <= buffer_stocks[p, d], 
                f"Station_{p}_Capacity"
            )

        # Energy consumption constraints
        for p in range(num_stations):
            model.addCons(
                quicksum(buffer_vars[(p, d)] for d in range(num_zones)) <= energy_capacities[p], 
                f"Station_{p}_EnergyConsumption"
            )

        for d in range(num_zones):
            model.addCons(
                quicksum(buffer_vars[(p, d)] for p in range(num_stations)) == zone_demand[d], 
                f"Zone_{d}_DemandSatisfaction"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_zones': 120,
        'num_stations': 50,
        'min_charging_cost': 900,
        'max_charging_cost': 2700,
        'min_station_cost': 90,
        'max_station_cost': 180,
        'min_energy_capacity': 225,
        'max_energy_capacity': 3000,
        'max_distance': 112,
        'min_equipment_cost': 150,
        'max_equipment_cost': 10000,
        'min_link_cost': 1050,
        'max_link_cost': 3000,
    }

    optimizer = EVChargingOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")