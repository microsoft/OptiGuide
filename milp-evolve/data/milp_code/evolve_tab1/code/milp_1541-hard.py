import random
import time
import numpy as np
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
        multiple_resource_costs = {'labor': hiring_costs,
                                   'equipment': np.random.randint(self.min_hiring_cost_equipment, self.max_hiring_cost_equipment + 1, self.n_sites)}

        project_costs = np.random.randint(self.min_project_cost, self.max_project_cost + 1, (self.n_sites, self.n_projects))
        capacities = np.random.randint(self.min_site_capacity, self.max_site_capacity + 1, self.n_sites)
        projects = np.random.gamma(2, 2, self.n_projects).astype(int) + 1

        penalty_costs = np.random.randint(self.min_penalty_cost, self.max_penalty_cost + 1, self.n_projects)

        graph = Graph.erdos_renyi(self.n_sites, self.link_probability, seed=self.seed)
        transportation_costs = np.random.randint(1, 20, (self.n_sites, self.n_sites))
        transportation_capacities = np.random.randint(self.transportation_capacity_min, self.transportation_capacity_max + 1, (self.n_sites, self.n_sites))

        return {
            "multiple_resource_costs": multiple_resource_costs,
            "project_costs": project_costs,
            "capacities": capacities,
            "projects": projects,
            "penalty_costs": penalty_costs,
            "graph": graph,
            "transportation_costs": transportation_costs,
            "transportation_capacities": transportation_capacities,
        }

    def solve(self, instance):
        multiple_resource_costs = instance['multiple_resource_costs']
        project_costs = instance['project_costs']
        capacities = instance['capacities']
        projects = instance['projects']
        penalty_costs = instance['penalty_costs']
        graph = instance['graph']
        transportation_costs = instance['transportation_costs']
        transportation_capacities = instance['transportation_capacities']

        model = Model("ConstructionResourceAllocation")
        n_sites = len(multiple_resource_costs['labor'])
        n_projects = len(project_costs[0])

        maintenance_vars = {c: model.addVar(vtype="B", name=f"MaintenanceTeam_Allocation_{c}") for c in range(n_sites)}
        project_vars = {(c, p): model.addVar(vtype="B", name=f"Site_{c}_Project_{p}") for c in range(n_sites) for p in range(n_projects)}
        transportation_vars = {(i, j): model.addVar(vtype="I", name=f"Transport_{i}_{j}") for i in range(n_sites) for j in range(n_sites) if i != j}
        penalty_vars = {p: model.addVar(vtype="B", name=f"Penalty_Project_{p}") for p in range(n_projects)}

        # Objective function
        model.setObjective(
            quicksum(multiple_resource_costs['labor'][c] * maintenance_vars[c] for c in range(n_sites)) +
            quicksum(multiple_resource_costs['equipment'][c] * maintenance_vars[c] for c in range(n_sites)) +
            quicksum(project_costs[c, p] * project_vars[c, p] for c in range(n_sites) for p in range(n_projects)) +
            quicksum(transportation_costs[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites) if i != j) +
            quicksum(penalty_costs[p] * penalty_vars[p] for p in range(n_projects)),
            "minimize"
        )

        # Constraints
        for p in range(n_projects):
            model.addCons(quicksum(project_vars[c, p] for c in range(n_sites)) == 1 - penalty_vars[p], f"Project_{p}_Allocation")

        for c in range(n_sites):
            for p in range(n_projects):
                model.addCons(project_vars[c, p] <= maintenance_vars[c], f"Site_{c}_Serve_{p}")

        for c in range(n_sites):
            model.addCons(quicksum(projects[p] * project_vars[c, p] for p in range(n_projects)) <= capacities[c], f"Site_{c}_Capacity")

        # Novel resource constraints, e.g., different limits for labor and equipment
        for resource in ['labor', 'equipment']:
            for c in range(n_sites):
                model.addCons(
                    quicksum(project_vars[c, p] for p in range(n_projects)) <= multiple_resource_costs[resource][c],
                    f"Site_{c}_Resource_{resource}_Capacity"
                )

        # Flow conservation at each site
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

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_sites': 48,
        'n_projects': 1680,
        'min_project_cost': 1652,
        'max_project_cost': 3000,
        'min_hiring_cost': 1397,
        'max_hiring_cost': 5000,
        'min_hiring_cost_equipment': 3000,
        'max_hiring_cost_equipment': 4500,
        'min_site_capacity': 2360,
        'max_site_capacity': 2874,
        'min_penalty_cost': 600,
        'max_penalty_cost': 2100,
        'link_probability': 0.74,
        'transportation_capacity_min': 2340,
        'transportation_capacity_max': 3000,
    }

    resource_optimizer = ConstructionResourceAllocation(parameters, seed=seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")