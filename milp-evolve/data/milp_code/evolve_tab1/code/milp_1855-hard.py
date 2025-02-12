import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MoreComplexEggDistributionOptimization:
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
        assert self.min_hub_cost >= 0 and self.max_hub_cost >= self.min_hub_cost
        assert self.min_link_cost >= 0 and self.max_link_cost >= self.min_link_cost
        assert self.min_hub_capacity > 0 and self.max_hub_capacity >= self.min_hub_capacity

        # Diversify cost generation using gamma distribution
        farm_costs = np.random.gamma(2, 2, self.num_farms) * (self.max_farm_cost - self.min_farm_cost) / 4 + self.min_farm_cost
        transport_costs = np.random.gamma(2, 2, (self.num_farms, self.num_markets)) * (self.max_transport_cost - self.min_transport_cost) / 4 + self.min_transport_cost
        egg_capacities = np.random.randint(self.min_egg_capacity, self.max_egg_capacity + 1, self.num_farms)
        market_demands = np.random.randint(1, 10, self.num_markets)
        distances = np.random.uniform(0, self.max_delivery_distance, (self.num_farms, self.num_markets))

        G = nx.DiGraph()
        node_pairs = []
        for p in range(self.num_farms):
            for d in range(self.num_markets):
                G.add_edge(f"farm_{p}", f"market_{d}")
                node_pairs.append((f"farm_{p}", f"market_{d}"))

        egg_varieties = 3
        variety_market_demands = np.random.randint(1, 10, (egg_varieties, self.num_markets))

        hub_activation_costs = np.random.randint(self.min_hub_cost, self.max_hub_cost + 1, self.num_farms)
        link_establishment_costs = np.random.randint(self.min_link_cost, self.max_link_cost + 1, (self.num_farms, self.num_markets))
        hub_capacities = np.random.randint(self.min_hub_capacity, self.max_hub_capacity + 1, self.num_farms)
        flow_capacities = np.random.randint(1, 50, (self.num_farms, self.num_markets))

        # Generating cliques for clique inequalities
        cliques = list(nx.find_cliques(G.to_undirected()))

        return {
            "farm_costs": farm_costs,
            "transport_costs": transport_costs,
            "egg_capacities": egg_capacities,
            "market_demands": market_demands,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs,
            "egg_varieties": egg_varieties,
            "variety_market_demands": variety_market_demands,
            "hub_activation_costs": hub_activation_costs,
            "link_establishment_costs": link_establishment_costs,
            "hub_capacities": hub_capacities,
            "flow_capacities": flow_capacities,
            "cliques": cliques,
        }

    def solve(self, instance):
        farm_costs = instance['farm_costs']
        transport_costs = instance['transport_costs']
        egg_capacities = instance['egg_capacities']
        market_demands = instance['market_demands']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']
        egg_varieties = instance['egg_varieties']
        variety_market_demands = instance['variety_market_demands']
        hub_activation_costs = instance['hub_activation_costs']
        link_establishment_costs = instance['link_establishment_costs']
        hub_capacities = instance['hub_capacities']
        flow_capacities = instance['flow_capacities']
        cliques = instance['cliques']

        model = Model("MoreComplexEggDistributionOptimization")
        num_farms = len(farm_costs)
        num_markets = len(transport_costs[0])

        # Decision variables
        egg_production_vars = {p: model.addVar(vtype="B", name=f"Farm_{p}") for p in range(num_farms)}
        shipment_vars = {(u, v): model.addVar(vtype="I", name=f"Shipment_{u}_{v}") for u, v in node_pairs}
        variety_farm_vars = {(vtype, u, v): model.addVar(vtype="C", name=f"Egg_{vtype}_{u}_{v}") for vtype in range(egg_varieties) for u, v in node_pairs}
        hub_activation_vars = {p: model.addVar(vtype="I", name=f"Hub_{p}") for p in range(num_farms)}
        link_activation_vars = {(p, d): model.addVar(vtype="B", name=f"Link_{p}_{d}") for p in range(num_farms) for d in range(num_markets)}
        flow_vars = {(p, d): model.addVar(vtype="I", name=f"Flow_{p}_{d}") for p in range(num_farms) for d in range(num_markets)}
        clique_activation_vars = {i: model.addVar(vtype="B", name=f"Clique_{i}") for i in range(len(cliques))}

        model.setObjective(
            quicksum(farm_costs[p] * egg_production_vars[p] for p in range(num_farms)) +
            quicksum(transport_costs[int(u.split('_')[1]), int(v.split('_')[1])] * shipment_vars[(u, v)] for (u, v) in node_pairs) +
            quicksum(hub_activation_costs[p] * hub_activation_vars[p] for p in range(num_farms)) +
            quicksum(link_establishment_costs[p, d] * link_activation_vars[p, d] for p in range(num_farms) for d in range(num_markets)) +
            quicksum(flow_vars[(p, d)] / flow_capacities[p, d] for p in range(num_farms) for d in range(num_markets)), 
            "minimize"
        )

        for d in range(num_markets):
            for vtype in range(egg_varieties):
                model.addCons(
                    quicksum(variety_farm_vars[(vtype, u, f"market_{d}")] for u in G.predecessors(f"market_{d}")) >= variety_market_demands[vtype, d], 
                    f"Market_{d}_NodeFlowConservation_{vtype}"
                )

        for p in range(num_farms):
            model.addCons(
                quicksum(variety_farm_vars[(vtype, f"farm_{p}", f"market_{d}")] for d in range(num_markets) for vtype in range(egg_varieties)) <= egg_capacities[p], 
                f"Farm_{p}_MaxEggCapacity"
            )

        for d in range(num_markets):
            model.addCons(
                quicksum(egg_production_vars[p] for p in range(num_farms) if distances[p, d] <= self.max_delivery_distance) >= 1, 
                f"Market_{d}_ZoneCoverage"
            )

        for u, v in node_pairs:
            model.addCons(
                shipment_vars[(u, v)] <= flow_capacities[int(u.split('_')[1]), int(v.split('_')[1])], 
                f"ShipmentCapacity_{u}_{v}"
            )

        for d in range(num_markets):
            model.addCons(
                quicksum(link_activation_vars[p, d] for p in range(num_farms)) == 1, 
                f"Market_{d}_HubAssignment"
            )

        for p in range(num_farms):
            for d in range(num_markets):
                model.addCons(
                    link_activation_vars[p, d] <= hub_activation_vars[p], 
                    f"Hub_{p}_Link_{d}"
                )

        for p in range(num_farms):
            model.addCons(
                quicksum(market_demands[d] * link_activation_vars[p, d] for d in range(num_markets)) <= hub_capacities[p] * hub_activation_vars[p], 
                f"Hub_{p}_Capacity"
            )
        
        for p in range(num_farms):
            model.addCons(
                quicksum(flow_vars[(p, d)] for d in range(num_markets)) <= egg_capacities[p], 
                f"Farm_{p}_FlowConservation"
            )
        
        for d in range(num_markets):
            model.addCons(
                quicksum(flow_vars[(p, d)] for p in range(num_farms)) == market_demands[d], 
                f"Market_{d}_FlowSatisfaction"
            )
        
        for p in range(num_farms):
            for d in range(num_markets):
                model.addCons(
                    flow_vars[(p, d)] <= flow_capacities[p, d], 
                    f"FlowCapacity_{p}_{d}"
                )

        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(hub_activation_vars[int(node.split('_')[1])] for node in clique) <= len(clique) * clique_activation_vars[i], 
                f"CliqueActivation_{i}"
            )
            model.addCons(
                clique_activation_vars[i] <= 1, 
                f"CliqueSingleActivation_{i}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_farms': 224,
        'num_markets': 50,
        'min_transport_cost': 450,
        'max_transport_cost': 450,
        'min_farm_cost': 450,
        'max_farm_cost': 900,
        'min_egg_capacity': 1800,
        'max_egg_capacity': 3000,
        'max_delivery_distance': 3000,
        'min_hub_cost': 3000,
        'max_hub_cost': 10000,
        'min_link_cost': 1225,
        'max_link_cost': 3000,
        'min_hub_capacity': 562,
        'max_hub_capacity': 750,
    }

    optimizer = MoreComplexEggDistributionOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")