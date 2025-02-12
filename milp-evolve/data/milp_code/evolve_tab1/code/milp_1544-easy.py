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
        project_costs = np.random.randint(self.min_project_cost, self.max_project_cost + 1, (self.n_sites, self.n_projects))
        capacities = np.random.randint(self.min_site_capacity, self.max_site_capacity + 1, self.n_sites)
        projects = np.random.gamma(2, 2, self.n_projects).astype(int) + 1

        graph = Graph.erdos_renyi(self.n_sites, self.link_probability, seed=self.seed)
        transportation_costs = np.random.randint(1, 20, (self.n_sites, self.n_sites))
        transportation_capacities = np.random.randint(self.transportation_capacity_min, self.transportation_capacity_max + 1, (self.n_sites, self.n_sites))

        battery_life = np.random.uniform(0.5, 2.0, self.n_sites)
        charging_stations = np.random.choice([0, 1], size=self.n_sites, p=[0.3, 0.7])
        project_deadlines = np.random.randint(1, 24, self.n_projects)
        penalty_costs = np.random.uniform(10, 50, self.n_projects)

        return {
            "hiring_costs": hiring_costs,
            "project_costs": project_costs,
            "capacities": capacities,
            "projects": projects,
            "graph": graph,
            "transportation_costs": transportation_costs,
            "transportation_capacities": transportation_capacities,
            "battery_life": battery_life,
            "charging_stations": charging_stations,
            "project_deadlines": project_deadlines,
            "penalty_costs": penalty_costs,
        }

    def solve(self, instance):
        hiring_costs = instance['hiring_costs']
        project_costs = instance['project_costs']
        capacities = instance['capacities']
        projects = instance['projects']
        graph = instance['graph']
        transportation_costs = instance['transportation_costs']
        transportation_capacities = instance['transportation_capacities']
        battery_life = instance['battery_life']
        charging_stations = instance['charging_stations']
        project_deadlines = instance['project_deadlines']
        penalty_costs = instance['penalty_costs']

        model = Model("ConstructionResourceAllocation")
        n_sites = len(hiring_costs)
        n_projects = len(project_costs[0])

        maintenance_vars = {c: model.addVar(vtype="B", name=f"MaintenanceTeam_Allocation_{c}") for c in range(n_sites)}
        project_vars = {(c, p): model.addVar(vtype="B", name=f"Site_{c}_Project_{p}") for c in range(n_sites) for p in range(n_projects)}
        transportation_vars = {(i, j): model.addVar(vtype="I", name=f"Transport_{i}_{j}") for i in range(n_sites) for j in range(n_sites)}

        battery_vars = {c: model.addVar(vtype="C", name=f"Battery_{c}") for c in range(n_sites)}
        penalty_vars = {p: model.addVar(vtype="C", name=f"Penalty_{p}") for p in range(n_projects)}
        charging_vars = {(c, t): model.addVar(vtype="B", name=f"Charge_{c}_{t}") for c in range(n_sites) for t in range(24)}

        # Objective function
        model.setObjective(
            quicksum(hiring_costs[c] * maintenance_vars[c] for c in range(n_sites)) +
            quicksum(project_costs[c, p] * project_vars[c, p] for c in range(n_sites) for p in range(n_projects)) +
            quicksum(transportation_costs[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(penalty_costs[p] * penalty_vars[p] for p in range(n_projects)),
            "minimize"
        )

        # Constraints
        for p in range(n_projects):
            model.addCons(quicksum(project_vars[c, p] for c in range(n_sites)) == 1, f"Project_{p}_Allocation")

        for c in range(n_sites):
            for p in range(n_projects):
                model.addCons(project_vars[c, p] <= maintenance_vars[c], f"Site_{c}_Serve_{p}")

        for c in range(n_sites):
            model.addCons(quicksum(projects[p] * project_vars[c, p] for p in range(n_projects)) <= capacities[c], f"Site_{c}_Capacity")

        for edge in graph.edges:
            model.addCons(maintenance_vars[edge[0]] + maintenance_vars[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

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
            model.addCons(battery_vars[c] <= battery_life[c], f"BatteryLife_{c}")
            model.addCons(quicksum(charging_vars[c, t] for t in range(24)) <= 24 * charging_stations[c], f"ChargingConstraint_{c}")

        for p in range(n_projects):
            model.addCons(quicksum(charging_vars[(c, t)] for c in range(n_sites) for t in range(project_deadlines[p])) >= project_vars[(c, p)], f"Deadline_{p}")
            model.addCons(battery_vars[c] + penalty_vars[p] >= project_deadlines[p], f"Penalty_{p}")

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_sites': 18,
        'n_projects': 126,
        'min_project_cost': 826,
        'max_project_cost': 3000,
        'min_hiring_cost': 1397,
        'max_hiring_cost': 5000,
        'min_site_capacity': 1180,
        'max_site_capacity': 1437,
        'link_probability': 0.17,
        'transportation_capacity_min': 1755,
        'transportation_capacity_max': 3000,
    }
    # Adjusted new parameters for added variables and constraints
    parameters.update({
        'battery_life_min': 0.5,
        'battery_life_max': 2.0,
        'charging_probability': 0.7,
    })

    resource_optimizer = ConstructionResourceAllocation(parameters, seed=seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")