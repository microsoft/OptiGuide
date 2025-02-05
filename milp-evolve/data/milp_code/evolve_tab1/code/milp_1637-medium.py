import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyMedicalSupplyOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_medical_stations > 0 and self.n_hospitals > 0
        assert self.min_station_cost >= 0 and self.max_station_cost >= self.min_station_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_station_capacity > 0 and self.max_station_capacity >= self.min_station_capacity

        station_costs = np.random.randint(self.min_station_cost, self.max_station_cost + 1, self.n_medical_stations)
        transport_costs_vehicle = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_medical_stations, self.n_hospitals))
        station_capacities = np.random.randint(self.min_station_capacity, self.max_station_capacity + 1, self.n_medical_stations)
        blood_unit_requirements = np.random.gamma(self.demand_shape, self.demand_scale, self.n_hospitals).astype(int)

        G = nx.DiGraph()
        node_pairs = []
        for m in range(self.n_medical_stations):
            for h in range(self.n_hospitals):
                G.add_edge(f"medical_station_{m}", f"hospital_{h}")
                node_pairs.append((f"medical_station_{m}", f"hospital_{h}"))

        return {
            "station_costs": station_costs,
            "transport_costs_vehicle": transport_costs_vehicle,
            "station_capacities": station_capacities,
            "blood_unit_requirements": blood_unit_requirements,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        station_costs = instance['station_costs']
        transport_costs_vehicle = instance['transport_costs_vehicle']
        station_capacities = instance['station_capacities']
        blood_unit_requirements = instance['blood_unit_requirements']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("EmergencyMedicalSupplyOptimization")
        n_medical_stations = len(station_costs)
        n_hospitals = len(transport_costs_vehicle[0])
        
        # Decision variables
        emergency_station_vars = {m: model.addVar(vtype="B", name=f"EmergencyStation_{m}") for m in range(n_medical_stations)}
        vehicle_transport_vars = {(u, v): model.addVar(vtype="C", name=f"VehicleTransport_{u}_{v}") for u, v in node_pairs}
        blood_unit_unmet_vars = {h: model.addVar(vtype="C", name=f"BloodUnitUnmet_{h}") for h in range(n_hospitals)}

        # Objective: minimize the total cost including medical station costs, transport costs, and unmet blood unit penalties.
        penalty_per_unit_unmet_demand = 2000
        model.setObjective(
            quicksum(station_costs[m] * emergency_station_vars[m] for m in range(n_medical_stations)) +
            quicksum(transport_costs_vehicle[m, int(v.split('_')[1])] * vehicle_transport_vars[(u, v)] for (u, v) in node_pairs for m in range(n_medical_stations) if u == f'medical_station_{m}') +
            penalty_per_unit_unmet_demand * quicksum(blood_unit_unmet_vars[h] for h in range(n_hospitals)),
            "minimize"
        )

        # Constraints: Ensure total blood unit supply matches demand accounting for unmet demand
        for h in range(n_hospitals):
            model.addCons(
                quicksum(vehicle_transport_vars[(u, f"hospital_{h}")] for u in G.predecessors(f"hospital_{h}")) + blood_unit_unmet_vars[h] >= blood_unit_requirements[h], 
                f"Hospital_{h}_BloodUnitRequirement"
            )

        # Constraints: Transport is feasible only if medical stations are operational and within emergency response time
        for m in range(n_medical_stations):
            for h in range(n_hospitals):
                model.addCons(
                    vehicle_transport_vars[(f"medical_station_{m}", f"hospital_{h}")] <= emergency_station_vars[m] * self.max_emergency_response_time,
                    f"MedicalStation_{m}_EmergencyResponseTime_{h}"
                )

        # Constraints: Medical stations cannot exceed their vehicle capacities
        for m in range(n_medical_stations):
            model.addCons(
                quicksum(vehicle_transport_vars[(f"medical_station_{m}", f"hospital_{h}")] for h in range(n_hospitals)) <= station_capacities[m], 
                f"MedicalStation_{m}_MaxStationCapacity"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_medical_stations': 1400,
        'n_hospitals': 32,
        'min_transport_cost': 432,
        'max_transport_cost': 810,
        'min_station_cost': 1012,
        'max_station_cost': 1579,
        'min_station_capacity': 2244,
        'max_station_capacity': 2250,
        'max_emergency_response_time': 1000,
        'demand_shape': 27.0,
        'demand_scale': 0.52,
    }

    optimizer = EmergencyMedicalSupplyOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")