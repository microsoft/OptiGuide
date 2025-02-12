import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum

class Graph:
    """
    Helper function: Container for a graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, edges_to_attach):
        """
        Generate a Barabási-Albert random graph.
        """
        edges = set()
        neighbors = {node: set() for node in range(number_of_nodes)}
        G = nx.barabasi_albert_graph(number_of_nodes, edges_to_attach)
        degrees = np.zeros(number_of_nodes, dtype=int)
        for edge in G.edges:
            edges.add(edge)
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class UrbanPlanningOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.edges_to_attach)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        public_service_demands = np.random.randint(1, 100, size=graph.number_of_nodes)
        budget_constraints = np.random.randint(1000, 5000, size=1)[0]
        service_center_installation_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        infrastructure_development_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 100
        transportation_links_exist = np.random.randint(0, 2, size=(graph.number_of_nodes, graph.number_of_nodes))
        
        # Urban planning specific data
        historical_site_importance = np.random.uniform(0.1, 1.0, size=graph.number_of_nodes)
        historical_preservation_costs = np.random.rand(graph.number_of_nodes) * 200
        social_inequality_factors = np.random.uniform(0.1, 1.0, size=graph.number_of_nodes)

        res = {
            'graph': graph,
            'public_service_demands': public_service_demands,
            'budget_constraints': budget_constraints,
            'service_center_installation_costs': service_center_installation_costs,
            'infrastructure_development_costs': infrastructure_development_costs,
            'transportation_links_exist': transportation_links_exist,
            'historical_site_importance': historical_site_importance,
            'historical_preservation_costs': historical_preservation_costs,
            'social_inequality_factors': social_inequality_factors
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        public_service_demands = instance['public_service_demands']
        budget_constraints = instance['budget_constraints']
        service_center_installation_costs = instance['service_center_installation_costs']
        infrastructure_development_costs = instance['infrastructure_development_costs']
        transportation_links_exist = instance['transportation_links_exist']
        historical_site_importance = instance['historical_site_importance']
        historical_preservation_costs = instance['historical_preservation_costs']
        social_inequality_factors = instance['social_inequality_factors']

        model = Model("UrbanPlanningOptimization")

        # Add variables
        service_center_vars = {node: model.addVar(vtype="B", name=f"ServiceCenterSelection_{node}") for node in graph.nodes}
        service_route_vars = {(i, j): model.addVar(vtype="B", name=f"service_route_{i}_{j}") 
                              for i in graph.nodes for j in graph.nodes if transportation_links_exist[i, j]}
        historical_preservation_vars = {node: model.addVar(vtype="C", name=f"HistoricalPreservation_{node}") 
                                        for node in graph.nodes}
        development_budget_vars = {node: model.addVar(vtype="C", name=f"DevelopmentBudget_{node}") for node in graph.nodes}
        
        # Capacity Constraints for service centers
        for center in graph.nodes:
            model.addCons(quicksum(public_service_demands[node] * service_route_vars[node, center] 
                                   for node in graph.nodes if (node, center) in service_route_vars) <= 
                          service_center_installation_costs[center], name=f"CapacityServiceCenter_{center}")

        # Connection Constraints of each node to one service center
        for node in graph.nodes:
            model.addCons(quicksum(service_route_vars[node, center] 
                                   for center in graph.nodes if (node, center) in service_route_vars) == 1, 
                          name=f"ConnectionService_{node}")

        # Ensure routing to opened service centers
        for node in graph.nodes:
            for center in graph.nodes:
                if (node, center) in service_route_vars:
                    model.addCons(service_route_vars[node, center] <= service_center_vars[center], 
                                  name=f"ServiceRoute_{node}_{center}")

        # Budget Constraints
        model.addCons(quicksum(development_budget_vars[node] for node in graph.nodes) <= budget_constraints, 
                      name=f"BudgetConstraint")

        # Historical Preservation Constraints
        for node in graph.nodes:
            model.addCons(historical_preservation_vars[node] <= historical_preservation_costs[node], 
                          name=f"HistoricalPreservation_{node}")

        # Social Inequality Constraints
        for node in graph.nodes:
            model.addCons(quicksum(service_route_vars[node, center] * social_inequality_factors[node] 
                                   for center in graph.nodes if (node, center) in service_route_vars) <= 1, 
                          name=f"SocialInequality_{node}")

        # Objective: Minimize total development cost while maximizing service coverage and preservation of historical sites
        total_service_cost = quicksum(service_center_vars[node] * service_center_installation_costs[node] 
                                      for node in graph.nodes)
        total_development_cost = quicksum(infrastructure_development_costs[i, j] * service_route_vars[i, j] 
                                          for i in graph.nodes for j in graph.nodes if (i, j) in service_route_vars)
        total_preservation_cost = quicksum(historical_preservation_vars[node] for node in graph.nodes)

        total_cost = total_service_cost + total_development_cost + total_preservation_cost
    
        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 50,
        'edge_probability': 0.42,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 35,
    }

    urban_planning_optimization = UrbanPlanningOptimization(parameters, seed=seed)
    instance = urban_planning_optimization.generate_instance()
    solve_status, solve_time = urban_planning_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")