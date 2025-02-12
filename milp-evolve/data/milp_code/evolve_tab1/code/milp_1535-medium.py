import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx
from networkx.algorithms.clique import find_cliques

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors
        self.cliques = list(find_cliques(nx.Graph(edges)))

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

        environmental_impact = np.random.uniform(0.1, 10, (self.n_sites, self.n_sites))
        penalty_costs = np.random.uniform(10, 50, self.n_sites)

        sensorPlacementCost = np.random.randint(8000, 30000, self.nSensors)
        operationalCost = np.random.uniform(300, 1200, self.nSensors)
        monitoringNodes = np.random.randint(10, 300, self.nMonitoringNodes)
        communicationDistances = np.abs(np.random.normal(loc=300, scale=100, size=(self.nSensors, self.nMonitoringNodes)))
        communicationDistances = np.where(communicationDistances > self.maxCommunicationRange, self.maxCommunicationRange, communicationDistances)
        communicationDistances = np.where(communicationDistances < 1, 1, communicationDistances)

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
            "environmental_impact": environmental_impact,
            "penalty_costs": penalty_costs,
            "sensorPlacementCost": sensorPlacementCost,
            "operationalCost": operationalCost,
            "monitoringNodes": monitoringNodes,
            "communicationDistances": communicationDistances,
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
        environmental_impact = instance['environmental_impact']
        penalty_costs = instance['penalty_costs']
        sensorPlacementCost = instance["sensorPlacementCost"]
        operationalCost = instance["operationalCost"]
        monitoringNodes = instance["monitoringNodes"]
        communicationDistances = instance["communicationDistances"]

        model = Model("ConstructionResourceAllocation")
        n_sites = len(hiring_costs)
        n_projects = len(project_costs[0])
        totalSensors = len(sensorPlacementCost)
        totalNodes = len(monitoringNodes)

        maintenance_vars = {c: model.addVar(vtype="B", name=f"MaintenanceTeam_Allocation_{c}") for c in range(n_sites)}
        project_vars = {(c, p): model.addVar(vtype="B", name=f"Site_{c}_Project_{p}") for c in range(n_sites) for p in range(n_projects)}
        resource_usage_vars = {(i, j): model.addVar(vtype="I", name=f"Resource_Usage_{i}_{j}") for i in range(n_sites) for j in range(n_sites)}
        transportation_vars = {(i, j): model.addVar(vtype="I", name=f"Transport_{i}_{j}") for i in range(n_sites) for j in range(n_sites)}

        sensor_vars = {s: model.addVar(vtype="B", name=f"Sensor_{s}") for s in range(totalSensors)}
        areaCoverage_vars = {(s, n): model.addVar(vtype="B", name=f"AreaCoverage_{s}_{n}") for s in range(totalSensors) for n in range(totalNodes)}

        # Objective function
        model.setObjective(
            quicksum(hiring_costs[c] * maintenance_vars[c] for c in range(n_sites)) +
            quicksum(project_costs[c, p] * project_vars[c, p] for c in range(n_sites) for p in range(n_projects)) +
            quicksum(transportation_costs[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(material_transport_costs[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(efficiency_costs[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(hiring_penalties[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(environmental_impact[i, j] * transportation_vars[i, j] for i in range(n_sites) for j in range(n_sites)) +
            quicksum(penalty_costs[i] for i in range(n_sites)) +
            quicksum(sensorPlacementCost[s] * sensor_vars[s] for s in range(totalSensors)) + 
            quicksum(communicationDistances[s, n] * areaCoverage_vars[s, n] for s in range(totalSensors) for n in range(totalNodes)),
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

        # Replacing edge constraints with clique constraints
        for clique in graph.cliques:
            if len(clique) > 1:
                model.addCons(quicksum(maintenance_vars[node] for node in clique) <= 1, f"Clique_{'_'.join(map(str, clique))}")

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

        for s in range(totalSensors):
            model.addCons(
                quicksum(communicationDistances[s, n] * areaCoverage_vars[s, n] for n in range(totalNodes)) <= self.maxCommunicationRange,
                f"CommunicationLimit_{s}"
            )

        for n in range(totalNodes):
            model.addCons(
                quicksum(areaCoverage_vars[s, n] for s in range(totalSensors)) >= 1,
                f"NodeCoverage_{n}"
            )

        for s in range(totalSensors):
            for n in range(totalNodes):
                model.addCons(
                    areaCoverage_vars[s, n] <= sensor_vars[s],
                    f"Coverage_{s}_{n}"
                )

        # Symmetry Breaking Constraints for sensors
        for s in range(totalSensors - 1):
            model.addCons(
                sensor_vars[s] >= sensor_vars[s + 1],
                f"Symmetry_{s}"
            )

        # Hierarchical Demand Fulfillment
        for n in range(totalNodes):
            for s1 in range(totalSensors):
                for s2 in range(s1 + 1, totalSensors):
                    model.addCons(
                        areaCoverage_vars[s2, n] <= areaCoverage_vars[s1, n] + int(communicationDistances[s1, n] <= self.maxCommunicationRange),
                        f"Hierarchical_{s1}_{s2}_{n}"
                    )

        # New Constraints: Frequency of Data Collection
        for s in range(totalSensors):
            model.addCons(
                quicksum(areaCoverage_vars[s, n] * monitoringNodes[n] for n in range(totalNodes)) >= self.dataFrequency,
                f"DataFrequency_{s}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_sites': 12,
        'n_projects': 420,
        'min_project_cost': 123,
        'max_project_cost': 3000,
        'min_hiring_cost': 69,
        'max_hiring_cost': 5000,
        'min_site_capacity': 118,
        'max_site_capacity': 2874,
        'link_probability': 0.66,
        'transportation_capacity_min': 1755,
        'transportation_capacity_max': 3000,
        'hiring_penalty_min': 0.73,
        'hiring_penalty_max': 45.0,
        'penalty_cost_min': 450,
        'penalty_cost_max': 562,
        'nSensors': 45,
        'nMonitoringNodes': 18,
        'budgetLimit': 100000,
        'maxCommunicationRange': 900,
        'dataFrequency': 500,
    }

    resource_optimizer = ConstructionResourceAllocation(parameters, seed=seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")