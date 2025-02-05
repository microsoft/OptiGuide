import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class CommunityCentersMILP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_city_graph(self):
        n_nodes = np.random.randint(self.min_zones, self.max_zones)
        G = nx.barabasi_albert_graph(n=n_nodes, m=self.market_connectivity, seed=self.seed)
        return G

    def generate_service_data(self, G):
        for node in G.nodes:
            G.nodes[node]['population'] = np.random.randint(500, 5000)
        for u, v in G.edges:
            G[u][v]['travel_time'] = np.random.randint(5, 50)
            G[u][v]['capacity'] = np.random.randint(50, 200)
            
    def create_neighborhoods(self, G):
        neighborhoods = list(nx.find_cliques(G))
        return neighborhoods

    def get_instance(self):
        G = self.generate_city_graph()
        self.generate_service_data(G)
        neighborhoods = self.create_neighborhoods(G)

        center_capacity = {node: np.random.randint(200, 1000) for node in G.nodes}
        travel_times = {(u, v): np.random.uniform(10.0, 100.0) for u, v in G.edges}
        daily_traffic = [(neighborhood, np.random.uniform(500, 3000)) for neighborhood in neighborhoods]

        operation_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            operation_scenarios[s]['population'] = {node: np.random.normal(G.nodes[node]['population'], G.nodes[node]['population'] * self.population_variation)
                                                    for node in G.nodes}
            operation_scenarios[s]['travel_time'] = {(u, v): np.random.normal(G[u][v]['travel_time'], G[u][v]['travel_time'] * self.time_variation)
                                                     for u, v in G.edges}
            operation_scenarios[s]['center_capacity'] = {node: np.random.normal(center_capacity[node], center_capacity[node] * self.capacity_variation)
                                                         for node in G.nodes}
        
        service_revenue = {node: np.random.uniform(100, 400) for node in G.nodes}
        operational_costs = {(u, v): np.random.uniform(20.0, 60.0) for u, v in G.edges}

        # New data for community centers
        n_centers = np.random.randint(self.center_min_count, self.center_max_count)
        comm_distance_matrix = np.random.uniform(5, 500, size=(n_centers, n_centers)).tolist()
        population_distribution = np.random.dirichlet(np.ones(3), size=n_centers).tolist()  # 3 demographic groups
        regional_costs = np.random.normal(loc=300, scale=50, size=n_centers).tolist()
        utility_costs = np.random.normal(loc=300, scale=50, size=n_centers).tolist()
        setup_cost = np.random.uniform(200, 1000, size=n_centers).tolist()
        
        # New operation data
        service_flow_costs = {(u, v): np.random.uniform(2, 30) for u, v in G.edges}

        return {
            'G': G,
            'neighborhoods': neighborhoods,
            'center_capacity': center_capacity,
            'travel_times': travel_times,
            'daily_traffic': daily_traffic,
            'piecewise_segments': self.piecewise_segments,
            'operation_scenarios': operation_scenarios,
            'service_revenue': service_revenue,
            'operational_costs': operational_costs,
            'n_centers': n_centers,
            'comm_distance_matrix': comm_distance_matrix,
            'population_distribution': population_distribution,
            'regional_costs': regional_costs,
            'utility_costs': utility_costs,
            'setup_cost': setup_cost,
            'service_flow_costs': service_flow_costs,
        }

    def solve(self, instance):
        G, neighborhoods = instance['G'], instance['neighborhoods']
        center_capacity = instance['center_capacity']
        travel_times = instance['travel_times']
        daily_traffic = instance['daily_traffic']
        piecewise_segments = instance['piecewise_segments']
        operation_scenarios = instance['operation_scenarios']
        service_revenue = instance['service_revenue']
        operational_costs = instance['operational_costs']
        n_centers = instance['n_centers']
        comm_distance_matrix = instance['comm_distance_matrix']
        population_distribution = instance['population_distribution']
        regional_costs = instance['regional_costs']
        utility_costs = instance['utility_costs']
        setup_cost = instance['setup_cost']
        service_flow_costs = instance['service_flow_costs']

        model = Model("CommunityCentersMILP")
        
        # Define variables
        zone_vars = {node: model.addVar(vtype="B", name=f"Zone{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        traffic_var = model.addVar(vtype="C", name="traffic_var")
        daily_traffic_vars = {i: model.addVar(vtype="B", name=f"Traffic_{i}") for i in range(len(daily_traffic))}
        segment_vars = {(u, v): {segment: model.addVar(vtype="B", name=f"Segment_{u}_{v}_{segment}") for segment in range(1, piecewise_segments + 1)} for u, v in G.edges}
        
        # New variables for diverse constraints and objectives
        utility_use_vars = {node: model.addVar(vtype="B", name=f"UtilityUse{node}") for node in G.nodes}
        community_vars = {node: model.addVar(vtype="I", name=f"Community{node}", lb=1, ub=5) for node in G.nodes}
        center_vars = {j: model.addVar(vtype="B", name=f"Center{j}") for j in range(n_centers)}
        travel_cost_vars = {j: model.addVar(vtype="C", name=f"CommTravelCost{j}") for j in range(n_centers)}
        demographic_vars = {(g, j): model.addVar(vtype="B", name=f"Demographic_{g}_{j}") for g in range(3) for j in range(n_centers)}
        regional_cost_vars = {j: model.addVar(vtype="C", name=f"RegionalCost{j}") for j in range(n_centers)}
        service_flow_vars = {f"ServiceFlow_{u}_{v}": model.addVar(vtype="C", name=f"ServiceFlow_{u}_{v}", lb=0) for u, v in G.edges}

        # Objective function
        objective_expr = quicksum(
            operation_scenarios[s]['population'][node] * zone_vars[node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            quicksum(segment * segment_vars[(u,v)][segment] for segment in range(1, piecewise_segments + 1))
            for u, v in G.edges
        )
        objective_expr -= quicksum(
            travel_times[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(price * daily_traffic_vars[i] for i, (neighborhood, price) in enumerate(daily_traffic))
        objective_expr -= quicksum(utility_use_vars[node] * 50 for node in G.nodes)
        objective_expr += quicksum(service_revenue[node] * zone_vars[node] for node in G.nodes)
        objective_expr -= quicksum(operational_costs[(u, v)] * route_vars[f"Route_{u}_{v}"] for u, v in G.edges)
        objective_expr -= quicksum(setup_cost[j] * center_vars[j] for j in range(n_centers))
        objective_expr -= quicksum(travel_cost_vars[j] for j in range(n_centers))
        objective_expr -= quicksum(regional_cost_vars[j] for j in range(n_centers))
        objective_expr -= quicksum(service_flow_costs[(u, v)] * service_flow_vars[f"ServiceFlow_{u}_{v}"] for u, v in G.edges)

        # Constraints
        for i, neighborhood in enumerate(neighborhoods):
            model.addCons(
                quicksum(zone_vars[node] for node in neighborhood) <= 1,
                name=f"ZoneGroup_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                zone_vars[u] + zone_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"ServiceFlow_{u}_{v}"
            )
            model.addCons(
                route_vars[f"Route_{u}_{v}"] == quicksum(segment_vars[(u, v)][segment] for segment in range(1, piecewise_segments + 1)),
                name=f"PiecewiseDist_{u}_{v}"
            )
            model.addCons(
                service_flow_vars[f"ServiceFlow_{u}_{v}"] <= G[u][v]['capacity'],
                name=f"Capacity_{u}_{v}"
            )
            
        model.addCons(
            traffic_var <= self.service_hours,
            name="ServiceTime_Limit"
        )

        # New Constraints to ensure balance and resource utilization
        for node in G.nodes:
            model.addCons(
                utility_use_vars[node] <= zone_vars[node],
                name=f"UtilityUseConstraint_{node}"
            )
            model.addCons(
                quicksum(segment_vars[(u, v)][segment] for u, v in G.edges if u == node or v == node for segment in range(1, piecewise_segments + 1)) <= community_vars[node] * 15,
                name=f"CommunityConstraint_{node}"
            )

        for j in range(n_centers):
            model.addCons(quicksum(comm_distance_matrix[j][k] * center_vars[k] for k in range(n_centers)) == travel_cost_vars[j], f"CommTravelCost_{j}")

        for g in range(3):
            for j in range(n_centers):
                model.addCons(demographic_vars[(g, j)] <= population_distribution[j][g] * center_vars[j], f"Demographic_{g}_{j}")

        for j in range(n_centers):
            model.addCons(regional_cost_vars[j] == regional_costs[j] * center_vars[j], f"RegionalCost_{j}")
        
        # Flow Continuity Constraints
        for node in G.nodes:
            inflow = quicksum(service_flow_vars[f"ServiceFlow_{u}_{v}"] for u, v in G.edges if v == node)
            outflow = quicksum(service_flow_vars[f"ServiceFlow_{u}_{v}"] for u, v in G.edges if u == node)
            model.addCons(inflow == outflow, name=f"ServiceFlow_Continuity_{node}")

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_zones': 150,
        'max_zones': 500,
        'market_connectivity': 10,
        'service_hours': 1800,
        'piecewise_segments': 30,
        'no_of_scenarios': 600,
        'population_variation': 0.3,
        'time_variation': 0.4,
        'capacity_variation': 0.35,
        'financial_param1': 1000,
        'financial_param2': 2000,
        'cost_param_1': 300.0,
        'operation_capacity': 1200.0,
        'center_min_count': 300,
        'center_max_count': 500,
    }

    community_milp = CommunityCentersMILP(parameters, seed=seed)
    instance = community_milp.get_instance()
    solve_status, solve_time = community_milp.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")