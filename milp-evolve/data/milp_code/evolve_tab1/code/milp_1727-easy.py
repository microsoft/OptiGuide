import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class PharmacyNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_pharmacies > 0 and self.n_hospitals > 0
        assert self.min_pharmacy_cost >= 0 and self.max_pharmacy_cost >= self.min_pharmacy_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_pharmacy_capacity > 0 and self.max_pharmacy_capacity >= self.min_pharmacy_capacity

        pharmacy_costs = np.random.randint(self.min_pharmacy_cost, self.max_pharmacy_cost + 1, self.n_pharmacies)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_pharmacies, self.n_hospitals))
        pharmacy_capacities = np.random.randint(self.min_pharmacy_capacity, self.max_pharmacy_capacity + 1, self.n_pharmacies)
        hospital_demand = np.random.randint(self.demand_mean - self.demand_range, self.demand_mean + self.demand_range + 1, self.n_hospitals)

        delivery_windows = np.random.randint(1, self.max_delivery_window + 1, (self.n_pharmacies, self.n_hospitals))
        quality_assurance = np.random.randint(1, self.max_quality_assurance + 1, (self.n_pharmacies, self.n_hospitals))
        emergency_deliveries = np.random.choice([0, 1], size=(self.n_pharmacies), p=[0.5, 0.5])

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.n_pharmacies):
            for h in range(self.n_hospitals):
                G.add_edge(f"pharmacy_{p}", f"hospital_{h}")
                node_pairs.append((f"pharmacy_{p}", f"hospital_{h}"))

        return {
            "pharmacy_costs": pharmacy_costs,
            "transport_costs": transport_costs,
            "pharmacy_capacities": pharmacy_capacities,
            "hospital_demand": hospital_demand,
            "delivery_windows": delivery_windows,
            "quality_assurance": quality_assurance,
            "emergency_deliveries": emergency_deliveries,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        pharmacy_costs = instance['pharmacy_costs']
        transport_costs = instance['transport_costs']
        pharmacy_capacities = instance['pharmacy_capacities']
        hospital_demand = instance['hospital_demand']
        delivery_windows = instance['delivery_windows']
        quality_assurance = instance['quality_assurance']
        emergency_deliveries = instance['emergency_deliveries']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("PharmacyNetworkOptimization")
        n_pharmacies = len(pharmacy_costs)
        n_hospitals = len(transport_costs[0])

        # Decision variables
        pharmacy_operational = {p: model.addVar(vtype="B", name=f"OperationalPharmacy_{p}") for p in range(n_pharmacies)}
        supply_transportation = {(u, v): model.addVar(vtype="C", name=f"SupplyTransport_{u}_{v}") for u, v in node_pairs}
        emergency_delivery = {(u, v): model.addVar(vtype="B", name=f"EmergencyDelivery_{u}_{v}") for u, v in node_pairs}
        quality_assurance_vars = {(u, v): model.addVar(vtype="C", name=f"QualityAssurance_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total distribution costs.
        model.setObjective(
            quicksum(pharmacy_costs[p] * pharmacy_operational[p] for p in range(n_pharmacies)) +
            quicksum(transport_costs[p, int(v.split('_')[1])] * supply_transportation[(u, v)] for (u, v) in node_pairs for p in range(n_pharmacies) if u == f'pharmacy_{p}'),
            "minimize"
        )

        # Constraints: Ensure total medical supplies match hospital demand
        for h in range(n_hospitals):
            model.addCons(
                quicksum(supply_transportation[(u, f"hospital_{h}")] for u in G.predecessors(f"hospital_{h}")) >= hospital_demand[h], 
                f"BasicSupplyDemand_{h}"
            )

        # Constraints: Transport is feasible only if pharmacies are operational and within delivery windows
        for p in range(n_pharmacies):
            for h in range(n_hospitals):
                model.addCons(
                    supply_transportation[(f"pharmacy_{p}", f"hospital_{h}")] <= pharmacy_operational[p] * self.max_delivery_window,
                    f"Pharmacy_{p}_MaxDeliveryWindow_{h}"
                )

        # Constraints: Supplies distribution should not exceed pharmacy capacities
        for p in range(n_pharmacies):
            model.addCons(
                quicksum(supply_transportation[(f"pharmacy_{p}", f"hospital_{h}")] for h in range(n_hospitals)) <= pharmacy_capacities[p], 
                f"EnoughCapacity_{p}"
            )

        # New Constraints: Delivery Time Window
        for u, v in node_pairs:
            p = int(u.split('_')[1])
            h = int(v.split('_')[1])
            model.addCons(
                supply_transportation[(u, v)] <= delivery_windows[p, h] * (1 - emergency_delivery[(u, v)]),
                f"DeliveryWindow_{u}_{v}"
            )

        # New Constraints: Quality Assurance
        for u, v in node_pairs:
            p = int(u.split('_')[1])
            h = int(v.split('_')[1])
            model.addCons(
                quality_assurance_vars[(u, v)] <= quality_assurance[p, h],
                f"VariableQualityAssurance_{u}_{v}"
            )

        # New Constraints: Emergency Delivery Option
        for p in range(n_pharmacies):
            for h in range(n_hospitals):
                model.addCons(
                    supply_transportation[(f"pharmacy_{p}", f"hospital_{h}")] <= emergency_deliveries[p] * self.max_delivery_window,
                    f"EmergencyDelivery_{p}_{h}"
                )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_pharmacies': 1500,
        'n_hospitals': 45,
        'min_transport_cost': 2250,
        'max_transport_cost': 3000,
        'min_pharmacy_cost': 1250,
        'max_pharmacy_cost': 3000,
        'min_pharmacy_capacity': 1500,
        'max_pharmacy_capacity': 2500,
        'max_delivery_window': 450,
        'demand_mean': 900,
        'demand_range': 800,
        'max_quality_assurance': 1000,
    }

    optimizer = PharmacyNetworkOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")