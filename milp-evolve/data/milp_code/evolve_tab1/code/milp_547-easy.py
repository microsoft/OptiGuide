import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MedicalSupplyDistribution:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_distribution_graph(self):
        n_nodes = np.random.randint(self.min_hospitals, self.max_hospitals)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.route_prob, directed=True, seed=self.seed)
        return G

    def generate_supply_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demands'] = np.random.randint(30, 200)
        for u, v in G.edges:
            G[u][v]['transit_time'] = np.random.randint(1, 5)
            G[u][v]['capacity'] = np.random.randint(10, 60)

    def generate_incompatibility_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.exclusion_rate:
                E_invalid.add(edge)
        return E_invalid

    def create_customer_groups(self, G):
        customers = list(nx.find_cliques(G.to_undirected()))
        return customers

    def get_instance(self):
        G = self.generate_distribution_graph()
        self.generate_supply_data(G)
        E_invalid = self.generate_incompatibility_data(G)
        customers = self.create_customer_groups(G)

        hospital_capacities = {node: np.random.randint(50, 300) for node in G.nodes}
        transportation_costs = {(u, v): np.random.uniform(20.0, 100.0) for u, v in G.edges}
        daily_customer_demands = [(group, np.random.uniform(100, 800)) for group in customers]

        customer_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            customer_scenarios[s]['demands'] = {node: np.random.normal(G.nodes[node]['demands'], G.nodes[node]['demands'] * self.demand_variation)
                                                for node in G.nodes}
            customer_scenarios[s]['transit_time'] = {(u, v): np.random.normal(G[u][v]['transit_time'], G[u][v]['transit_time'] * self.time_variation)
                                                     for u, v in G.edges}
            customer_scenarios[s]['hospital_capacities'] = {node: np.random.normal(hospital_capacities[node], hospital_capacities[node] * self.cap_variation)
                                                            for node in G.nodes}
        
        nutritional_requirements = {node: np.random.uniform(30, 120) for node in G.nodes}
        transportation_cost = {(u, v): np.random.uniform(10.0, 70.0) for u, v in G.edges}

        hospital_count = np.random.randint(self.hospital_min_count, self.hospital_max_count)
        travel_distance_matrix = np.random.uniform(20, 500, size=(hospital_count, hospital_count)).tolist()
        disease_statistics = np.random.dirichlet(np.ones(3), size=hospital_count).tolist()  # 3 disease groups
        regional_supply_costs = np.random.normal(loc=50, scale=10, size=hospital_count).tolist()
        operational_cost = np.random.uniform(50, 200, size=hospital_count).tolist()
        
        return {
            'G': G,
            'E_invalid': E_invalid,
            'customers': customers,
            'hospital_capacities': hospital_capacities,
            'transportation_costs': transportation_costs,
            'daily_customer_demands': daily_customer_demands,
            'piecewise_segments': self.piecewise_segments,
            'customer_scenarios': customer_scenarios,
            'nutritional_requirements': nutritional_requirements,
            'transportation_cost': transportation_cost,
            'hospital_count': hospital_count,
            'travel_distance_matrix': travel_distance_matrix,
            'disease_statistics': disease_statistics,
            'regional_supply_costs': regional_supply_costs,
            'operational_cost': operational_cost,
        }

    def solve(self, instance):
        G, E_invalid, customers = instance['G'], instance['E_invalid'], instance['customers']
        hospital_capacities = instance['hospital_capacities']
        transportation_costs = instance['transportation_costs']
        daily_customer_demands = instance['daily_customer_demands']
        piecewise_segments = instance['piecewise_segments']
        customer_scenarios = instance['customer_scenarios']
        nutritional_requirements = instance['nutritional_requirements']
        transportation_cost = instance['transportation_cost']
        hospital_count = instance['hospital_count']
        travel_distance_matrix = instance['travel_distance_matrix']
        disease_statistics = instance['disease_statistics']
        regional_supply_costs = instance['regional_supply_costs']
        operational_cost = instance['operational_cost']

        model = Model("MedicalSupplyDistribution")

        # Define variables
        supply_vars = {node: model.addVar(vtype="B", name=f"Supply{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        supply_budget = model.addVar(vtype="C", name="supply_budget")
        daily_demand_vars = {i: model.addVar(vtype="B", name=f"Demand_{i}") for i in range(len(daily_customer_demands))}
        segment_vars = {(u, v): {segment: model.addVar(vtype="B", name=f"Segment_{u}_{v}_{segment}") for segment in range(1, piecewise_segments + 1)} for u, v in G.edges}

        # New variables for diverse constraints and objectives
        nutrition_vars = {node: model.addVar(vtype="C", name=f"NutriReq{node}") for node in G.nodes}
        disease_group_vars = {node: model.addVar(vtype="I", name=f"DiseaseGroup{node}", lb=1, ub=3) for node in G.nodes}
        hospital_vars = {j: model.addVar(vtype="B", name=f"Hospital{j}") for j in range(hospital_count)}
        travel_cost_vars = {j: model.addVar(vtype="C", name=f"TravelCost{j}") for j in range(hospital_count)}
        disease_stat_vars = {(d, j): model.addVar(vtype="B", name=f"DiseaseStat_{d}_{j}") for d in range(3) for j in range(hospital_count)}
        supply_cost_vars = {j: model.addVar(vtype="C", name=f"SupplyCost{j}") for j in range(hospital_count)}

        # Objective function
        objective_expr = quicksum(
            customer_scenarios[s]['demands'][node] * supply_vars[node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            quicksum(segment * segment_vars[(u,v)][segment] for segment in range(1, piecewise_segments + 1))
            for u, v in E_invalid
        )
        objective_expr -= quicksum(
            transportation_costs[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(demand * daily_demand_vars[i] for i, (group, demand) in enumerate(daily_customer_demands))
        objective_expr -= quicksum(nutrition_vars[node] * 10 for node in G.nodes)
        objective_expr += quicksum(nutritional_requirements[node] * supply_vars[node] for node in G.nodes)
        objective_expr -= quicksum(transportation_cost[(u, v)] * route_vars[f"Route_{u}_{v}"] for u, v in G.edges)
        objective_expr -= quicksum(operational_cost[j] * hospital_vars[j] for j in range(hospital_count))
        objective_expr -= quicksum(travel_cost_vars[j] for j in range(hospital_count))
        objective_expr -= quicksum(supply_cost_vars[j] for j in range(hospital_count))

        # Constraints
        for i, customer in enumerate(customers):
            model.addCons(
                quicksum(supply_vars[node] for node in customer) <= 1,
                name=f"SupplyGroup_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                supply_vars[u] + supply_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"Flow_{u}_{v}"
            )
            model.addCons(
                route_vars[f"Route_{u}_{v}"] == quicksum(segment_vars[(u, v)][segment] for segment in range(1, piecewise_segments + 1)),
                name=f"PiecewiseTransit_{u}_{v}"
            )
            
        model.addCons(
            supply_budget <= self.supply_hours,
            name="SupplyTime_Limit"
        )

        # New Constraints to ensure balance and resource utilization
        for node in G.nodes:
            model.addCons(
                nutrition_vars[node] >= supply_vars[node] * nutritional_requirements[node],
                name=f"NutritionConstraint_{node}"
            )
            model.addCons(
                quicksum(segment_vars[(u, v)][segment] for u, v in G.edges if u == node or v == node for segment in range(1, piecewise_segments + 1)) <= disease_group_vars[node] * 15,
                name=f"DiseaseGroupConstraint_{node}"
            )

        for j in range(hospital_count):
            model.addCons(quicksum(travel_distance_matrix[j][k] * hospital_vars[k] for k in range(hospital_count)) == travel_cost_vars[j], f"TravelCost_{j}")

        for d in range(3):
            for j in range(hospital_count):
                model.addCons(disease_stat_vars[(d, j)] <= disease_statistics[j][d] * hospital_vars[j], f"DiseaseStat_{d}_{j}")

        for j in range(hospital_count):
            model.addCons(supply_cost_vars[j] == regional_supply_costs[j] * hospital_vars[j], f"SupplyCost_{j}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_hospitals': 7,
        'max_hospitals': 175,
        'route_prob': 0.17,
        'exclusion_rate': 0.24,
        'supply_hours': 1000,
        'piecewise_segments': 30,
        'no_of_scenarios': 600,
        'demand_variation': 0.8,
        'time_variation': 0.8,
        'cap_variation': 0.73,
        'financial_param1': 800,
        'financial_param2': 1500,
        'transportation_cost_param_1': 300.0,
        'move_capacity': 750.0,
        'hospital_min_count': 100,
        'hospital_max_count': 3000,
    }

    msd = MedicalSupplyDistribution(parameters, seed=seed)
    instance = msd.get_instance()
    solve_status, solve_time = msd.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")