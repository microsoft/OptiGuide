import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class VaccineDistributionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_production_zones > 0 and self.n_distribution_zones > 0
        assert self.min_zone_cost >= 0 and self.max_zone_cost >= self.min_zone_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_production_capacity > 0 and self.max_production_capacity >= self.min_production_capacity
        assert self.max_transport_distance >= 0

        zone_costs = np.random.randint(self.min_zone_cost, self.max_zone_cost + 1, self.n_production_zones)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_production_zones, self.n_distribution_zones))
        production_capacities = np.random.randint(self.min_production_capacity, self.max_production_capacity + 1, self.n_production_zones)
        distribution_demands = np.random.randint(1, 10, self.n_distribution_zones)
        budget_limits = np.random.uniform(self.min_budget_limit, self.max_budget_limit, self.n_production_zones)
        distances = np.random.uniform(0, self.max_transport_distance, (self.n_production_zones, self.n_distribution_zones))
        shipment_costs = np.random.randint(1, 20, (self.n_production_zones, self.n_distribution_zones))
        shipment_capacities = np.random.randint(5, 15, (self.n_production_zones, self.n_distribution_zones))
        inventory_holding_costs = np.random.normal(self.holding_cost_mean, self.holding_cost_sd, self.n_distribution_zones)
        demand_levels = np.random.normal(self.demand_mean, self.demand_sd, self.n_distribution_zones)

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.n_production_zones):
            for d in range(self.n_distribution_zones):
                G.add_edge(f"production_{p}", f"distribution_{d}")
                node_pairs.append((f"production_{p}", f"distribution_{d}"))

        # New instance data
        equipment_use = {p: np.random.randint(0, 2) for p in range(self.n_production_zones)}
        patient_groups = {p: np.random.randint(1, 5) for p in range(self.n_production_zones)}
        
        vaccine_types = 3  # Different types of vaccines
        type_by_zone = np.random.randint(1, vaccine_types + 1, self.n_production_zones)
        type_transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (vaccine_types, self.n_production_zones, self.n_distribution_zones))
        type_distribution_demands = np.random.randint(1, 10, (vaccine_types, self.n_distribution_zones))
        type_inventory_holding_costs = np.random.normal(self.holding_cost_mean, self.holding_cost_sd, (vaccine_types, self.n_distribution_zones))
        delivery_times = np.random.randint(1, self.max_delivery_time, (self.n_production_zones, self.n_distribution_zones))

        return {
            "zone_costs": zone_costs,
            "transport_costs": transport_costs,
            "production_capacities": production_capacities,
            "distribution_demands": distribution_demands,
            "budget_limits": budget_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "shipment_costs": shipment_costs,
            "shipment_capacities": shipment_capacities,
            "inventory_holding_costs": inventory_holding_costs,
            "demand_levels": demand_levels,
            "equipment_use": equipment_use,
            "patient_groups": patient_groups,
            "vaccine_types": vaccine_types,
            "type_by_zone": type_by_zone,
            "type_transport_costs": type_transport_costs,
            "type_distribution_demands": type_distribution_demands,
            "type_inventory_holding_costs": type_inventory_holding_costs,
            "delivery_times": delivery_times
        }

    def solve(self, instance):
        zone_costs = instance['zone_costs']
        transport_costs = instance['transport_costs']
        production_capacities = instance['production_capacities']
        distribution_demands = instance['distribution_demands']
        budget_limits = instance['budget_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        shipment_costs = instance['shipment_costs']
        shipment_capacities = instance['shipment_capacities']
        inventory_holding_costs = instance['inventory_holding_costs']
        demand_levels = instance['demand_levels']
        equipment_use = instance['equipment_use']
        patient_groups = instance['patient_groups']
        vaccine_types = instance['vaccine_types']
        type_by_zone = instance['type_by_zone']
        type_transport_costs = instance['type_transport_costs']
        type_distribution_demands = instance['type_distribution_demands']
        type_inventory_holding_costs = instance['type_inventory_holding_costs']
        delivery_times = instance['delivery_times']

        model = Model("VaccineDistributionOptimization")
        n_production_zones = len(zone_costs)
        n_distribution_zones = len(transport_costs[0])

        # Decision variables
        vaccine_vars = {p: model.addVar(vtype="B", name=f"Production_{p}") for p in range(n_production_zones)}
        shipment_vars = {(u, v): model.addVar(vtype="I", name=f"Shipment_{u}_{v}") for u, v in node_pairs}
        inventory_level = {f"inv_{d + 1}": model.addVar(vtype="C", name=f"inv_{d + 1}") for d in range(n_distribution_zones)}
        equip_vars = {p: model.addVar(vtype="B", name=f"EquipUse_{p}") for p in range(n_production_zones)}
        patient_groups_vars = {p: model.addVar(vtype="I", name=f"PatientGroup_{p}", lb=1, ub=5) for p in range(n_production_zones)}
        type_zone_vars = {(vtype, u, v): model.addVar(vtype="C", name=f"Vaccine_{vtype}_{u}_{v}") for vtype in range(vaccine_types) for u, v in node_pairs}

        # Objective: minimize the total cost including production zone costs, transport costs, shipment costs, and inventory holding costs.
        model.setObjective(
            quicksum(zone_costs[p] * vaccine_vars[p] for p in range(n_production_zones)) +
            quicksum(type_transport_costs[vtype, p, int(v.split('_')[1])] * type_zone_vars[(vtype, u, v)] for vtype in range(vaccine_types) for (u, v) in node_pairs for p in range(n_production_zones) if u == f'production_{p}') +
            quicksum(shipment_costs[int(u.split('_')[1]), int(v.split('_')[1])] * shipment_vars[(u, v)] for (u, v) in node_pairs) +
            quicksum(type_inventory_holding_costs[vtype, d] * inventory_level[f"inv_{d + 1}"] for vtype in range(vaccine_types) for d in range(n_distribution_zones)),
            "minimize"
        )

        # Vaccine distribution constraint for each zone
        for d in range(n_distribution_zones):
            for vtype in range(vaccine_types):
                model.addCons(
                    quicksum(type_zone_vars[(vtype, u, f"distribution_{d}")] for u in G.predecessors(f"distribution_{d}")) == type_distribution_demands[vtype, d], 
                    f"Distribution_{d}_NodeFlowConservation_vtype_{vtype}"
                )

        # Constraints: Zones only receive vaccines if the production zones are operational
        for p in range(n_production_zones):
            for d in range(n_distribution_zones):
                for vtype in range(vaccine_types):
                    model.addCons(
                        type_zone_vars[(vtype, f"production_{p}", f"distribution_{d}")] <= budget_limits[p] * vaccine_vars[p], 
                        f"Production_{p}_VaccineLimitByBudget_{d}_vtype_{vtype}"
                    )

        # Constraints: Production zones cannot exceed their vaccine production capacities
        for p in range(n_production_zones):
            for vtype in range(vaccine_types):
                model.addCons(
                    quicksum(type_zone_vars[(vtype, f"production_{p}", f"distribution_{d}")] for d in range(n_distribution_zones)) <= production_capacities[p], 
                    f"Production_{p}_MaxZoneCapacity_vtype_{vtype}"
                )

        # Coverage constraint for Elderly zones
        for d in range(n_distribution_zones):
            model.addCons(
                quicksum(vaccine_vars[p] for p in range(n_production_zones) if distances[p, d] <= self.max_transport_distance) >= 1, 
                f"Distribution_{d}_ElderyZoneCoverage"
            )

        # Constraints: Shipment cannot exceed its capacity
        for u, v in node_pairs:
            model.addCons(shipment_vars[(u, v)] <= shipment_capacities[int(u.split('_')[1]), int(v.split('_')[1])], f"ShipmentCapacity_{u}_{v}")

        # Inventory Constraints
        for d in range(n_distribution_zones):
            for vtype in range(vaccine_types):
                model.addCons(
                    inventory_level[f"inv_{d + 1}"] - demand_levels[d] >= 0,
                    f"Stock_{d + 1}_out_vtype_{vtype}"
                )

        # New Constraints: Ensure equipment use and align transport with patient groups
        for p in range(n_production_zones):
            model.addCons(equip_vars[p] <= vaccine_vars[p], f"Equip_Use_Constraint_{p}")
            for d in range(n_distribution_zones):
                model.addCons(quicksum(shipment_vars[(f"production_{p}", f"distribution_{d}")] for d in range(n_distribution_zones)) <= patient_groups_vars[p] * 10, f"Group_Transport_Constraint_{p}_{d}")

        # New Constraints: Time delivery constraints
        for p in range(n_production_zones):
            for d in range(n_distribution_zones):
                for vtype in range(vaccine_types):
                    model.addCons(delivery_times[p, d] * type_zone_vars[(vtype, f"production_{p}", f"distribution_{d}")] <= self.max_delivery_time, f"Delivery_Time_Constraint_{p}_{d}_vtype_{vtype}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_production_zones': 393,
        'n_distribution_zones': 18,
        'min_transport_cost': 48,
        'max_transport_cost': 270,
        'min_zone_cost': 900,
        'max_zone_cost': 1579,
        'min_production_capacity': 40,
        'max_production_capacity': 140,
        'min_budget_limit': 45,
        'max_budget_limit': 1201,
        'max_transport_distance': 1062,
        'holding_cost_mean': 300.0,
        'holding_cost_sd': 1.25,
        'demand_mean': 168.8,
        'demand_sd': 0.56,
        'equipment_use_threshold': 21,
        'patient_groups_limit': 2,
        'max_delivery_time': 2,
    }

    optimizer = VaccineDistributionOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")