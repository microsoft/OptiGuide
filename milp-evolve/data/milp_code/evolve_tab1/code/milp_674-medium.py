import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HealthcareSupplyOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_network_graph(self):
        n_nodes = np.random.randint(self.min_centers, self.max_centers)
        G = nx.watts_strogatz_graph(n=n_nodes, k=self.small_world_k, p=self.small_world_p, seed=self.seed)
        return G

    def generate_node_load_data(self, G):
        for node in G.nodes:
            G.nodes[node]['loads'] = np.random.randint(50, 500)
        for u, v in G.edges:
            G[u][v]['dist_time'] = np.random.randint(1, 5)
            G[u][v]['cap'] = np.random.randint(20, 100)

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def get_instance(self):
        G = self.generate_network_graph()
        self.generate_node_load_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        cliques = self.create_cliques(G)

        center_cap = {node: np.random.randint(100, 500) for node in G.nodes}
        dist_cost = {(u, v): np.random.uniform(10.0, 50.0) for u, v in G.edges}
        daily_distributions = [(i, np.random.uniform(500, 2000)) for i in range(len(cliques))]

        dist_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            dist_scenarios[s]['loads'] = {node: np.random.normal(G.nodes[node]['loads'], G.nodes[node]['loads'] * self.load_variation)
                                          for node in G.nodes}
            dist_scenarios[s]['dist_time'] = {(u, v): np.random.normal(G[u][v]['dist_time'], G[u][v]['dist_time'] * self.time_variation)
                                              for u, v in G.edges}
            dist_scenarios[s]['center_cap'] = {node: np.random.normal(center_cap[node], center_cap[node] * self.cap_variation)
                                               for node in G.nodes}

        financial_rewards = {node: np.random.uniform(50, 200) for node in G.nodes}
        travel_costs = {(u, v): np.random.uniform(5.0, 30.0) for u, v in G.edges}

        n_facilities = np.random.randint(self.facility_min_count, self.facility_max_count)
        travel_distance_matrix = np.random.uniform(10, 1000, size=(n_facilities, n_facilities)).tolist()
        demographic_data = np.random.dirichlet(np.ones(5), size=n_facilities).tolist()  # 5 demographic groups
        regional_supply_prices = np.random.normal(loc=100, scale=20, size=n_facilities).tolist()
        energy_consumption = np.random.normal(loc=100, scale=20, size=n_facilities).tolist()
        setup_cost = np.random.uniform(100, 500, size=n_facilities).tolist()

        flow_costs = {(u, v): np.random.uniform(1, 15) for u, v in G.edges}

        return {
            'G': G,
            'E_invalid': E_invalid,
            'cliques': cliques,
            'center_cap': center_cap,
            'dist_cost': dist_cost,
            'daily_distributions': daily_distributions,
            'piecewise_segments': self.piecewise_segments,
            'dist_scenarios': dist_scenarios,
            'financial_rewards': financial_rewards,
            'travel_costs': travel_costs,
            'n_facilities': n_facilities,
            'travel_distance_matrix': travel_distance_matrix,
            'demographic_data': demographic_data,
            'regional_supply_prices': regional_supply_prices,
            'energy_consumption': energy_consumption,
            'setup_cost': setup_cost,
            'flow_costs': flow_costs,
        }

    def solve(self, instance):
        G, E_invalid, cliques = instance['G'], instance['E_invalid'], instance['cliques']
        center_cap = instance['center_cap']
        dist_cost = instance['dist_cost']
        daily_distributions = instance['daily_distributions']
        piecewise_segments = instance['piecewise_segments']
        dist_scenarios = instance['dist_scenarios']
        financial_rewards = instance['financial_rewards']
        travel_costs = instance['travel_costs']
        n_facilities = instance['n_facilities']
        travel_distance_matrix = instance['travel_distance_matrix']
        demographic_data = instance['demographic_data']
        regional_supply_prices = instance['regional_supply_prices']
        energy_consumption = instance['energy_consumption']
        setup_cost = instance['setup_cost']
        flow_costs = instance['flow_costs']

        model = Model("HealthcareSupplyOptimization")

        # Define variables
        carrier_vars = {node: model.addVar(vtype="B", name=f"Carrier{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        dist_budget = model.addVar(vtype="C", name="dist_budget")
        daily_dist_vars = {i: model.addVar(vtype="B", name=f"Dist_{i}") for i in range(len(daily_distributions))}
        segment_vars = {(u, v): {segment: model.addVar(vtype="B", name=f"Segment_{u}_{v}_{segment}") for segment in range(1, piecewise_segments + 1)} for u, v in G.edges}

        equipment_use_vars = {node: model.addVar(vtype="B", name=f"EquipUse{node}") for node in G.nodes}
        header_vars = {node: model.addVar(vtype="I", name=f"Header{node}", lb=1, ub=5) for node in G.nodes}
        facility_vars = {j: model.addVar(vtype="B", name=f"Facility{j}") for j in range(n_facilities)}
        travel_cost_vars = {j: model.addVar(vtype="C", name=f"TravelCost{j}") for j in range(n_facilities)}
        demographic_vars = {(g, j): model.addVar(vtype="B", name=f"Demographic_{g}_{j}") for g in range(5) for j in range(n_facilities)}
        supply_cost_vars = {j: model.addVar(vtype="C", name=f"SupplyCost{j}") for j in range(n_facilities)}
        flow_vars = {f"Flow_{u}_{v}": model.addVar(vtype="C", name=f"Flow_{u}_{v}", lb=0) for u, v in G.edges}

        # Objective function
        objective_expr = quicksum(
            dist_scenarios[s]['loads'][node] * carrier_vars[node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            quicksum(segment * segment_vars[(u, v)][segment] for segment in range(1, piecewise_segments + 1))
            for u, v in E_invalid
        )
        objective_expr -= quicksum(
            dist_cost[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(price * daily_dist_vars[i] for i, (bundle, price) in enumerate(daily_distributions))
        objective_expr -= quicksum(equipment_use_vars[node] * 20 for node in G.nodes)
        objective_expr += quicksum(financial_rewards[node] * carrier_vars[node] for node in G.nodes)
        objective_expr -= quicksum(travel_costs[(u, v)] * route_vars[f"Route_{u}_{v}"] for u, v in G.edges)
        objective_expr -= quicksum(setup_cost[j] * facility_vars[j] for j in range(n_facilities))
        objective_expr -= quicksum(travel_cost_vars[j] for j in range(n_facilities))
        objective_expr -= quicksum(supply_cost_vars[j] for j in range(n_facilities))
        objective_expr -= quicksum(flow_costs[(u, v)] * flow_vars[f"Flow_{u}_{v}"] for u, v in G.edges)

        # Constraints
        for clique_index, clique in enumerate(cliques):
            model.addCons(
                quicksum(carrier_vars[node] for node in clique) <= 1,
                name=f"CarrierClique_{clique_index}"
            )

        for u, v in G.edges:
            model.addCons(
                carrier_vars[u] + carrier_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"Flow_{u}_{v}"
            )
            model.addCons(
                route_vars[f"Route_{u}_{v}"] == quicksum(segment_vars[(u, v)][segment] for segment in range(1, piecewise_segments + 1)),
                name=f"PiecewiseDist_{u}_{v}"
            )
            model.addCons(
                flow_vars[f"Flow_{u}_{v}"] <= G[u][v]['cap'],
                name=f"Cap_{u}_{v}"
            )

        # Distribution budget constraint
        model.addCons(
            dist_budget <= self.dist_hours,
            name="DistTime_Limit"
        )

        # New Constraints to ensure balance and resource utilization
        for node in G.nodes:
            model.addCons(
                equipment_use_vars[node] <= carrier_vars[node],
                name=f"EquipUseConstraint_{node}"
            )
            model.addCons(
                quicksum(segment_vars[(u, v)][segment] for u, v in G.edges if u == node or v == node for segment in range(1, piecewise_segments + 1)) <= header_vars[node] * 20,
                name=f"HeaderConstraint_{node}"
            )

        for j in range(n_facilities):
            model.addCons(
                quicksum(travel_distance_matrix[j][k] * facility_vars[k] for k in range(n_facilities)) == travel_cost_vars[j],
                f"TravelCost_{j}"
            )

        for g in range(5):
            for j in range(n_facilities):
                model.addCons(
                    demographic_vars[(g, j)] <= demographic_data[j][g] * facility_vars[j],
                    f"Demographic_{g}_{j}"
                )

        for j in range(n_facilities):
            model.addCons(
                supply_cost_vars[j] == regional_supply_prices[j] * facility_vars[j],
                f"SupplyCost_{j}"
            )

        # Flow Continuity Constraints
        for node in G.nodes:
            inflow = quicksum(flow_vars[f"Flow_{u}_{v}"] for u, v in G.edges if v == node)
            outflow = quicksum(flow_vars[f"Flow_{u}_{v}"] for u, v in G.edges if u == node)
            model.addCons(inflow == outflow, name=f"Flow_Continuity_{node}")

        # Energy Consumption Constraint
        model.addCons(
            quicksum(energy_consumption[j] * facility_vars[j] for j in range(n_facilities)) <= self.energy_limit,
            name="EnergyConsumptionLimit"
        )        
        
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_centers': 220,
        'max_centers': 675,
        'route_prob': 0.74,
        'exclusion_rate': 0.8,
        'dist_hours': 2361,
        'piecewise_segments': 45,
        'no_of_scenarios': 945,
        'load_variation': 0.6,
        'time_variation': 0.61,
        'cap_variation': 0.59,
        'facility_min_count': 840,
        'facility_max_count': 945,
        'small_world_k': 12,
        'small_world_p': 0.24,
        'energy_limit': 10000,
    }

    hco = HealthcareSupplyOptimization(parameters, seed=seed)
    instance = hco.get_instance()
    solve_status, solve_time = hco.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")