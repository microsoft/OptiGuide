import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, probability, seed=None):
        graph = nx.erdos_renyi_graph(number_of_nodes, probability, seed=seed)
        edges = set(graph.edges)
        degrees = [d for (n, d) in graph.degree]
        neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes}
        return Graph(number_of_nodes, edges, degrees, neighbors)

class ConstructionResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        hiring_costs = np.random.randint(self.min_hiring_cost, self.max_hiring_cost + 1, self.n_sites)
        project_costs = np.random.randint(self.min_project_cost, self.max_project_cost + 1, (self.n_sites, self.n_projects))
        capacities = np.random.randint(self.min_site_capacity, self.max_site_capacity + 1, self.n_sites)
        projects = np.random.gamma(2, 2, self.n_projects).astype(int) + 1

        graph = Graph.erdos_renyi(self.n_sites, self.link_probability, seed=self.seed)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))

        transportation_costs = np.random.randint(1, 20, (self.n_sites, self.n_sites))
        transportation_capacities = np.random.randint(self.transportation_capacity_min, self.transportation_capacity_max + 1, (self.n_sites, self.n_sites))

        material_transport_costs = np.random.randint(10, 50, (self.n_sites, self.n_sites))
        efficiency_costs = np.random.randint(5, 30, (self.n_sites, self.n_sites))
        hiring_penalties = np.random.uniform(low=0.5, high=2.0, size=(self.n_sites, self.n_sites))

        cross_border_lower_bounds = np.random.randint(1, 5, (self.n_sites, self.n_sites))
        cross_border_upper_bounds = np.random.randint(10, 20, (self.n_sites, self.n_sites))
        cross_border_penalties = np.random.randint(1, 10, (self.n_sites, self.n_sites))

        compliance_costs = np.random.randint(10, 1000, self.n_projects)
        
        environmental_impact = np.random.uniform(0.1, 10, (self.n_sites, self.n_sites))

        return {
            "hiring_costs": hiring_costs,
            "project_costs": project_costs,
            "capacities": capacities,
            "projects": projects,
            "graph": graph,
            "edge_weights": edge_weights,
            "transportation_costs": transportation_costs,
            "transportation_capacities": transportation_capacities,
            "material_transport_costs": material_transport_costs,
            "efficiency_costs": efficiency_costs,
            "hiring_penalties": hiring_penalties,
            "cross_border_lower_bounds": cross_border_lower_bounds,
            "cross_border_upper_bounds": cross_border_upper_bounds,
            "cross_border_penalties": cross_border_penalties,
            "compliance_costs": compliance_costs,
            "environmental_impact": environmental_impact
        }

    def solve(self, instance):
        hiring_costs = instance['hiring_costs']
        project_costs = instance['project_costs']
        capacities = instance['capacities']
        projects = instance['projects']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        transportation_costs = instance['transportation_costs']
        transportation_capacities = instance['transportation_capacities']
        material_transport_costs = instance['material_transport_costs']
        efficiency_costs = instance['efficiency_costs']
        hiring_penalties = instance['hiring_penalties']
        cross_border_lower_bounds = instance['cross_border_lower_bounds']
        cross_border_upper_bounds = instance['cross_border_upper_bounds']
        cross_border_penalties = instance['cross_border_penalties']
        compliance_costs = instance['compliance_costs']
        environmental_impact = instance['environmental_impact']

        model = Model("ConstructionResourceAllocation")
        n_sites = len(hiring_costs)
        n_projects = len(project_costs[0])

        maintenance_vars = {c: model.addVar(vtype="B", name=f"MaintenanceTeam_Allocation_{c}") for c in range(n_sites)}
        project_vars = {(c, p): model.addVar(vtype="B", name=f"Site_{c}_Project_{p}") for c in range(n_sites) for p in range(n_projects)}
        resource_usage_vars = {(i, j): model.addVar(vtype="I", name=f"Resource_Usage_{i}_{j}") for i in range(n_sites) for j in range(n_sites)}
        transportation_vars = {(i, j): model.addVar(vtype="I", name=f"Transport_{i}_{j}") for i in range(n_sites) for j in range(n_sites)}

        cross_border_usage_vars = {(i, j): model.addVar(vtype="C", name=f"CrossBorder_Usage_{i}_{j}", lb=cross_border_lower_bounds[i, j], ub=cross_border_upper_bounds[i, j]) for i in range(n_sites) for j in range(n_sites)}

        depot_vars = {i: model.addVar(vtype="B", name=f"Depot_{i}") for i in range(n_sites)}

        model.setObjective(
            quicksum(hiring_costs[c] * maintenance_vars[c] for c in range(n_sites)) +
            quicksum(project_costs[c, p] * project_vars[c, p] for c in range(n_sites) for p in range(n_projects)) +
            quicksum(transportation_costs[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(material_transport_costs[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(efficiency_costs[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(hiring_penalties[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(cross_border_penalties[i, j] * cross_border_usage_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(compliance_costs[p] * project_vars[c, p] for c in range(n_sites) for p in range(n_projects)) +
            quicksum(environmental_impact[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)),
            "minimize"
        )

        for p in range(n_projects):
            model.addCons(quicksum(project_vars[c, p] for c in range(n_sites)) == 1, f"Project_{p}_Allocation")

        for c in range(n_sites):
            for p in range(n_projects):
                model.addCons(project_vars[c, p] <= maintenance_vars[c], f"Site_{c}_Serve_{p}")

        for c in range(n_sites):
            model.addCons(quicksum(projects[p] * project_vars[c, p] for p in range(n_projects)) <= capacities[c], f"Site_{c}_Capacity")

        for edge in graph.edges:
            model.addCons(maintenance_vars[edge[0]] + maintenance_vars[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for i in range(n_sites):
            model.addCons(
                quicksum(resource_usage_vars[i, j] for j in range(n_sites) if i != j) ==
                quicksum(resource_usage_vars[j, i] for j in range(n_sites) if i != j),
                f"Usage_Conservation_{i}"
            )

        for j in range(n_sites):
            model.addCons(
                quicksum(transportation_vars[i, j] for i in range(n_sites) if i != j) ==
                quicksum(project_vars[j, p] for p in range(n_projects)),
                f"Transport_Conservation_{j}"
            )

        for i in range(n_sites):
            for j in range(n_sites):
                if i != j:
                    model.addCons(transportation_vars[i, j] <= transportation_capacities[i, j], f"Transport_Capacity_{i}_{j}")

        for c in range(n_sites):
            model.addCons(
                quicksum(project_vars[c, p] for p in range(n_projects)) <= n_projects * maintenance_vars[c],
                f"Convex_Hull_{c}"
            )

        for i in range(n_sites):
            for j in range(n_sites):
                if i != j:
                    model.addCons(cross_border_usage_vars[i, j] == resource_usage_vars[i, j], f"CrossBorder_Usage_{i}_{j}")

        for i in range(n_sites):
            model.addCons(
                quicksum(transportation_vars[i, j] for j in range(n_sites)) <= (self.depot_capacity*depot_vars[i]),
                f"Depot_Capacity_{i}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_sites': 66,
        'n_projects': 84,
        'min_project_cost': 1653,
        'max_project_cost': 3000,
        'min_hiring_cost': 1863,
        'max_hiring_cost': 5000,
        'min_site_capacity': 475,
        'max_site_capacity': 1919,
        'link_probability': 0.31,
        'transportation_capacity_min': 1875,
        'transportation_capacity_max': 3000,
        'hiring_penalty_min': 0.59,
        'hiring_penalty_max': 1200.0,
        'depot_capacity': 10000,
    }

    resource_optimizer = ConstructionResourceAllocation(parameters, seed=seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")