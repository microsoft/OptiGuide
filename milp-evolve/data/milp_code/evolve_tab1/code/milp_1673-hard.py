import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EggDistributionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.num_farms > 0 and self.num_markets > 0
        assert self.min_farm_cost >= 0 and self.max_farm_cost >= self.min_farm_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_egg_capacity > 0 and self.max_egg_capacity >= self.min_egg_capacity
        assert self.max_delivery_distance >= 0

        farm_costs = np.random.randint(self.min_farm_cost, self.max_farm_cost + 1, self.num_farms)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.num_farms, self.num_markets))
        egg_capacities = np.random.randint(self.min_egg_capacity, self.max_egg_capacity + 1, self.num_farms)
        market_demands = np.random.randint(1, 10, self.num_markets)
        budget_limits = np.random.uniform(self.min_budget_limit, self.max_budget_limit, self.num_farms)
        distances = np.random.uniform(0, self.max_delivery_distance, (self.num_farms, self.num_markets))
        shipment_costs = np.random.randint(1, 20, (self.num_farms, self.num_markets))
        shipment_capacities = np.random.randint(5, 15, (self.num_farms, self.num_markets))
        inventory_holding_costs = np.random.normal(self.holding_cost_mean, self.holding_cost_sd, self.num_markets)
        demand_levels = np.random.normal(self.demand_mean, self.demand_sd, self.num_markets)

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.num_farms):
            for d in range(self.num_markets):
                G.add_edge(f"farm_{p}", f"market_{d}")
                node_pairs.append((f"farm_{p}", f"market_{d}"))

        # New instance data
        equipment_use = {p: np.random.randint(0, 2) for p in range(self.num_farms)}
        breeding_costs = np.random.randint(50, 200, self.num_farms)
        
        egg_varieties = 3  # Different varieties of eggs
        variety_by_farm = np.random.randint(1, egg_varieties + 1, self.num_farms)
        variety_transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (egg_varieties, self.num_farms, self.num_markets))
        variety_market_demands = np.random.randint(1, 10, (egg_varieties, self.num_markets))
        variety_inventory_holding_costs = np.random.normal(self.holding_cost_mean, self.holding_cost_sd, (egg_varieties, self.num_markets))
        delivery_times = np.random.randint(1, self.max_delivery_time, (self.num_farms, self.num_markets))

        return {
            "farm_costs": farm_costs,
            "transport_costs": transport_costs,
            "egg_capacities": egg_capacities,
            "market_demands": market_demands,
            "breeding_costs": breeding_costs,
            "budget_limits": budget_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "shipment_costs": shipment_costs,
            "shipment_capacities": shipment_capacities,
            "inventory_holding_costs": inventory_holding_costs,
            "demand_levels": demand_levels,
            "equipment_use": equipment_use,
            "egg_varieties": egg_varieties,
            "variety_by_farm": variety_by_farm,
            "variety_transport_costs": variety_transport_costs,
            "variety_market_demands": variety_market_demands,
            "variety_inventory_holding_costs": variety_inventory_holding_costs,
            "delivery_times": delivery_times
        }

    def solve(self, instance):
        farm_costs = instance['farm_costs']
        transport_costs = instance['transport_costs']
        egg_capacities = instance['egg_capacities']
        market_demands = instance['market_demands']
        breeding_costs = instance['breeding_costs']
        budget_limits = instance['budget_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        shipment_costs = instance['shipment_costs']
        shipment_capacities = instance['shipment_capacities']
        inventory_holding_costs = instance['inventory_holding_costs']
        demand_levels = instance['demand_levels']
        equipment_use = instance['equipment_use']
        egg_varieties = instance['egg_varieties']
        variety_by_farm = instance['variety_by_farm']
        variety_transport_costs = instance['variety_transport_costs']
        variety_market_demands = instance['variety_market_demands']
        variety_inventory_holding_costs = instance['variety_inventory_holding_costs']
        delivery_times = instance['delivery_times']

        model = Model("EggDistributionOptimization")
        num_farms = len(farm_costs)
        num_markets = len(transport_costs[0])

        # Decision variables
        egg_production_vars = {p: model.addVar(vtype="B", name=f"Farm_{p}") for p in range(num_farms)}
        shipment_vars = {(u, v): model.addVar(vtype="I", name=f"Shipment_{u}_{v}") for u, v in node_pairs}
        inventory_level = {f"inv_{d + 1}": model.addVar(vtype="C", name=f"inv_{d + 1}") for d in range(num_markets)}
        equip_vars = {p: model.addVar(vtype="B", name=f"EquipUse_{p}") for p in range(num_farms)}
        variety_farm_vars = {(vtype, u, v): model.addVar(vtype="C", name=f"Egg_{vtype}_{u}_{v}") for vtype in range(egg_varieties) for u, v in node_pairs}

        # Objective: minimize the total cost including farm costs, transport costs, shipment costs, and inventory holding costs.
        model.setObjective(
            quicksum(farm_costs[p] * egg_production_vars[p] for p in range(num_farms)) +
            quicksum(variety_transport_costs[vtype, p, int(v.split('_')[1])] * variety_farm_vars[(vtype, u, v)] for vtype in range(egg_varieties) for (u, v) in node_pairs for p in range(num_farms) if u == f'farm_{p}') +
            quicksum(shipment_costs[int(u.split('_')[1]), int(v.split('_')[1])] * shipment_vars[(u, v)] for (u, v) in node_pairs) +
            quicksum(variety_inventory_holding_costs[vtype, d] * inventory_level[f"inv_{d + 1}"] for vtype in range(egg_varieties) for d in range(num_markets)),
            "minimize"
        )

        # Egg distribution constraint for each market
        for d in range(num_markets):
            for vtype in range(egg_varieties):
                model.addCons(
                    quicksum(variety_farm_vars[(vtype, u, f"market_{d}")] for u in G.predecessors(f"market_{d}")) == variety_market_demands[vtype, d], 
                    f"Market_{d}_NodeFlowConservation_vtype_{vtype}"
                )

        # Constraints: Markets only receive eggs if the farms are operational
        for p in range(num_farms):
            for d in range(num_markets):
                for vtype in range(egg_varieties):
                    model.addCons(
                        variety_farm_vars[(vtype, f"farm_{p}", f"market_{d}")] <= budget_limits[p] * egg_production_vars[p], 
                        f"Farm_{p}_EggLimitByBudget_{d}_vtype_{vtype}"
                    )

        # Constraints: Farms cannot exceed their egg production capacities
        for p in range(num_farms):
            for vtype in range(egg_varieties):
                model.addCons(
                    quicksum(variety_farm_vars[(vtype, f"farm_{p}", f"market_{d}")] for d in range(num_markets)) <= egg_capacities[p], 
                    f"Farm_{p}_MaxEggCapacity_vtype_{vtype}"
                )

        # Coverage constraint for all markets
        for d in range(num_markets):
            model.addCons(
                quicksum(egg_production_vars[p] for p in range(num_farms) if distances[p, d] <= self.max_delivery_distance) >= 1, 
                f"Market_{d}_ZoneCoverage"
            )

        # Constraints: Shipment cannot exceed its capacity
        for u, v in node_pairs:
            model.addCons(shipment_vars[(u, v)] <= shipment_capacities[int(u.split('_')[1]), int(v.split('_')[1])], f"ShipmentCapacity_{u}_{v}")

        # Inventory Constraints
        for d in range(num_markets):
            for vtype in range(egg_varieties):
                model.addCons(
                    inventory_level[f"inv_{d + 1}"] - demand_levels[d] >= 0,
                    f"Stock_{d + 1}_out_vtype_{vtype}"
                )

        # New Constraints: Ensure equipment use and handle breeding costs
        for p in range(num_farms):
            model.addCons(equip_vars[p] <= egg_production_vars[p], f"EquipUse_Constraint_{p}")
            model.addCons(breeding_costs[p] * egg_production_vars[p] <= self.max_breeding_cost, f"BreedingCost_Constraint_{p}")

        # New Constraints: Delivery constraints focusing on transport time
        for p in range(num_farms):
            for d in range(num_markets):
                for vtype in range(egg_varieties):
                    model.addCons(delivery_times[p, d] * variety_farm_vars[(vtype, f"farm_{p}", f"market_{d}")] <= self.max_delivery_time, f"Delivery_Time_Constraint_{p}_{d}_vtype_{vtype}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_farms': 100,
        'num_markets': 40,
        'min_transport_cost': 100,
        'max_transport_cost': 1000,
        'min_farm_cost': 400,
        'max_farm_cost': 1200,
        'min_egg_capacity': 400,
        'max_egg_capacity': 1500,
        'min_budget_limit': 1000,
        'max_budget_limit': 5000,
        'max_delivery_distance': 2000,
        'holding_cost_mean': 80.0,
        'holding_cost_sd': 15.0,
        'demand_mean': 1000.0,
        'demand_sd': 120.0,
        'max_breeding_cost': 1050,
        'max_delivery_time': 175,
    }

    optimizer = EggDistributionOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")