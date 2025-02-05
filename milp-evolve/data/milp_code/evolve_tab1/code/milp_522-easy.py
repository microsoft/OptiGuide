import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ComplexTeamDeployment:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        n_teams = random.randint(self.min_teams, self.max_teams)
        n_locations = random.randint(self.min_locations, self.max_locations)

        # Cost matrices
        team_costs = np.random.gamma(20, 2, size=(n_teams, n_locations))
        activation_costs = np.random.normal(100, 30, size=n_locations)

        # Capacities and demands
        transportation_capacity = np.random.randint(50, 200, size=n_locations)
        team_demand = np.random.randint(10, 25, size=n_teams)

        # Preferences and skill levels
        team_preferences = np.random.uniform(0, 1, size=(n_teams, n_locations))
        team_skills = np.random.randint(1, 10, size=n_teams)
        location_skill_requirements = np.random.randint(5, 15, size=n_locations)

        # Semi-Continuous bounds
        min_transport_scale = np.random.randint(10, 50, size=n_locations)
        
        ### new instance data code starts here
        
        # Generate a more complex graph for flow
        graph = nx.barabasi_albert_graph(n_locations, self.num_edges_per_node, seed=self.seed)
        capacities = np.random.gamma(shape=self.capacity_shape, scale=self.capacity_scale, size=len(graph.edges))
        flows = np.random.uniform(0, self.max_flow, size=len(graph.edges))

        # Generate start and end nodes for flow network
        source_node, sink_node = 0, n_locations - 1

        # Convert graph to adjacency list
        adj_list = {i: [] for i in range(n_locations)}
        for idx, (u, v) in enumerate(graph.edges):
            adj_list[u].append((v, flows[idx], capacities[idx]))
            adj_list[v].append((u, flows[idx], capacities[idx]))  # for undirected edges

        ### new instance data code ends here
        
        res = {
            'n_teams': n_teams,
            'n_locations': n_locations,
            'team_costs': team_costs,
            'activation_costs': activation_costs,
            'transportation_capacity': transportation_capacity,
            'team_demand': team_demand,
            'team_preferences': team_preferences,
            'team_skills': team_skills,
            'location_skill_requirements': location_skill_requirements,
            'min_transport_scale': min_transport_scale,
            'adj_list': adj_list,
            'source_node': source_node,
            'sink_node': sink_node
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        n_teams = instance['n_teams']
        n_locations = instance['n_locations']
        team_costs = instance['team_costs']
        activation_costs = instance['activation_costs']
        transportation_capacity = instance['transportation_capacity']
        team_demand = instance['team_demand']
        team_preferences = instance['team_preferences']
        team_skills = instance['team_skills']
        location_skill_requirements = instance['location_skill_requirements']
        min_transport_scale = instance['min_transport_scale']
        
        adj_list = instance['adj_list']
        source_node = instance['source_node']
        sink_node = instance['sink_node']

        model = Model("ComplexTeamDeployment")

        # Variables
        x = {}
        for i in range(n_teams):
            for j in range(n_locations):
                x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        z = {j: model.addVar(vtype="B", name=f"z_{j}") for j in range(n_locations)}
        transport_scale = {j: model.addVar(vtype="INTEGER", lb=min_transport_scale[j], ub=transportation_capacity[j], name=f"transport_scale_{j}") for j in range(n_locations)}
        
        ### new variables code starts here

        flow_vars = {}
        # Create flow variables for edges
        for u in adj_list:
            for v, _, capacity in adj_list[u]:
                flow_vars[(u, v)] = model.addVar(vtype='C', lb=0, ub=capacity, name=f"f_{u}_{v}")

        ### new variables code ends here

        # Objective function: Minimize total cost and maximize preferences
        total_cost = quicksum(x[i, j] * team_costs[i, j] for i in range(n_teams) for j in range(n_locations)) + \
                     quicksum(z[j] * activation_costs[j] for j in range(n_locations))
        total_preference = quicksum(x[i, j] * team_preferences[i, j] for i in range(n_teams) for j in range(n_locations))
        
        ### new objective code starts here

        cost_term = total_cost
        flow_term = quicksum(flow_vars[(u, v)] for u, v in flow_vars)
        objective_expr = cost_term - self.flow_weight * flow_term + total_preference
        
        model.setObjective(objective_expr, "minimize")

        ### new objective code ends here

        # Constraints
        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) <= 3, name=f"team_max_assignments_{i}")  # Max 3 locations

        for j in range(n_locations):
            model.addCons(quicksum(x[i, j] * team_demand[i] for i in range(n_teams)) <= transport_scale[j] * z[j], name=f"transportation_capacity_{j}")
            model.addCons(quicksum(x[i, j] * team_skills[i] for i in range(n_teams)) >= location_skill_requirements[j] * z[j], name=f"skill_requirement_{j}")

        for i in range(n_teams):
            model.addCons(quicksum(x[i, j] for j in range(n_locations)) >= 1, name=f"team_min_assignment_{i}")
        
        ### new constraints code starts here
        
        # Add flow constraints
        for node in adj_list:
            if node == source_node:
                model.addCons(quicksum(flow_vars[(source_node, v)] for v, _, _ in adj_list[source_node]) >= self.min_flow, f"flow_source_{source_node}")
            elif node == sink_node:
                model.addCons(quicksum(flow_vars[(u, sink_node)] for u, _, _ in adj_list[sink_node]) >= self.min_flow, f"flow_sink_{sink_node}")
            else:
                inflow = quicksum(flow_vars[(u, node)] for u, _, _ in adj_list[node] if (u, node) in flow_vars)
                outflow = quicksum(flow_vars[(node, v)] for v, _, _ in adj_list[node] if (node, v) in flow_vars)
                model.addCons(inflow - outflow == 0, f"flow_balance_{node}")

        ### new constraints code ends here

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_teams': 50,
        'max_teams': 120,
        'min_locations': 105,
        'max_locations': 2500,
        'max_assignments_per_team': 1215,
        'min_skill_level': 0,
        'max_skill_level': 90,
        'min_skill_requirement': 45,
        'max_skill_requirement': 220,
        'num_edges_per_node': 70,
        'capacity_shape': 1.12,
        'capacity_scale': 70,
        'max_flow': 18,
        'flow_weight': 0.1,
        'min_flow': 0,
    }

    deployment = ComplexTeamDeployment(parameters, seed=seed)
    instance = deployment.generate_instance()
    solve_status, solve_time = deployment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")