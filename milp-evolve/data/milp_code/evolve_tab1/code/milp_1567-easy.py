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

class HumanitarianAidDistribution:
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

    def get_instance(self):
        graph = self.generate_graph()
        critical_care_demand = np.random.randint(1, 100, size=graph.number_of_nodes)
        network_capacity = np.random.randint(100, 500, size=graph.number_of_nodes)
        zoning_costs = np.random.randint(50, 150, size=graph.number_of_nodes)
        transportation_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        service_hours = np.random.randint(1, 10, size=(graph.number_of_nodes, graph.number_of_nodes))

        access_restrictions = np.random.choice([0, 1], size=graph.number_of_nodes)
        security_risks = np.random.normal(loc=5.0, scale=2.0, size=graph.number_of_nodes)
        priority_safety = np.random.choice([0, 1], size=graph.number_of_nodes)
        theft_risks = np.random.gamma(shape=2.0, scale=1.0, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Auction-based allocation data
        n_bundles = 100
        bundles = []
        prices = np.random.randint(200, 500, size=n_bundles)
        items = np.arange(graph.number_of_nodes)
        for _ in range(n_bundles):
            bundle = np.random.choice(items, size=np.random.randint(1, graph.number_of_nodes//2 + 1), replace=False)
            bundles.append(bundle)

        # EMS-inspired data
        renewable_energy_costs = np.random.randint(50, 500, (graph.number_of_nodes, graph.number_of_nodes))
        carbon_emissions = np.random.randint(1, 10, (graph.number_of_nodes, graph.number_of_nodes))
        hazardous_material_penalty = np.random.randint(100, 1000, graph.number_of_nodes)

        segments = 3
        piecewise_points = np.linspace(0, 1, segments + 1)
        piecewise_slopes = np.random.rand(segments) * 10

        res = {
            'graph': graph,
            'critical_care_demand': critical_care_demand,
            'network_capacity': network_capacity,
            'zoning_costs': zoning_costs,
            'transportation_costs': transportation_costs,
            'service_hours': service_hours,
            'access_restrictions': access_restrictions,
            'security_risks': security_risks,
            'priority_safety': priority_safety,
            'theft_risks': theft_risks,
            'bundles': bundles,
            'prices': prices,
            'renewable_energy_costs': renewable_energy_costs,
            'carbon_emissions': carbon_emissions,
            'hazardous_material_penalty': hazardous_material_penalty,
            'piecewise_points': piecewise_points,
            'piecewise_slopes': piecewise_slopes,
            'segments': segments
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        critical_care_demand = instance['critical_care_demand']
        network_capacity = instance['network_capacity']
        zoning_costs = instance['zoning_costs']
        transportation_costs = instance['transportation_costs']
        service_hours = instance['service_hours']
        access_restrictions = instance['access_restrictions']
        security_risks = instance['security_risks']
        priority_safety = instance['priority_safety']
        theft_risks = instance['theft_risks']
        bundles = instance['bundles']
        prices = instance['prices']
        renewable_energy_costs = instance['renewable_energy_costs']
        carbon_emissions = instance['carbon_emissions']
        hazardous_material_penalty = instance['hazardous_material_penalty']
        piecewise_points = instance['piecewise_points']
        piecewise_slopes = instance['piecewise_slopes']
        segments = instance['segments']

        model = Model("HumanitarianAidDistribution")

        # Add variables
        MedicalCenterSelection_vars = {node: model.addVar(vtype="B", name=f"MedicalCenterSelection_{node}") for node in graph.nodes}
        HealthcareRouting_vars = {(i, j): model.addVar(vtype="B", name=f"HealthcareRouting_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        
        # New service time variables
        ServiceTime_vars = {node: model.addVar(vtype="C", name=f"ServiceTime_{node}") for node in graph.nodes}

        # New delivery time window satisfaction variables
        DeliveryWindowSatisfaction_vars = {node: model.addVar(vtype="B", name=f"DeliveryWindowSatisfaction_{node}") for node in graph.nodes}

        # Resource Theft Risk Variables
        ResourceTheftRisk_vars = {(i, j): model.addVar(vtype="C", name=f"ResourceTheftRisk_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        # Auction-based Resource Allocation Variables
        ResourceAllocation_vars = {i: model.addVar(vtype="B", name=f"ResourceAllocation_{i}") for i in range(len(bundles))}

        # Renewable energy and carbon emission variables
        RenewableEnergy_vars = {(i, j): model.addVar(vtype="B", name=f"RenewableEnergy_{i}_{j}") for i in graph.nodes for j in graph.nodes}

        ResponseTimePiecewise_vars = {(i, j, k): model.addVar(vtype="C", name=f"ResponseTimePiecewise_{i}_{j}_{k}") for i in graph.nodes for j in graph.nodes for k in range(segments)}
        EnergyPiecewise_vars = {(i, j, k) : model.addVar(vtype="C",name=f"EnergyPiecewise_{i}_{j}_{k}") for i in graph.nodes for j in graph.nodes for k in range(segments)}
        EmissionsPiecewise_vars = {(i,j,k) : model.addVar(vtype="C", name=f"EmissionsPiecewise_{i}_{j}_{k}") for i in graph.nodes for j in graph.nodes for k in range(segments)}

        # Network Capacity Constraints for healthcare centers
        for center in graph.nodes:
            model.addCons(quicksum(critical_care_demand[node] * HealthcareRouting_vars[node, center] for node in graph.nodes) <= network_capacity[center], name=f"NetworkCapacity_{center}")

        # Connection Constraints of each node to one healthcare center
        for node in graph.nodes:
            model.addCons(quicksum(HealthcareRouting_vars[node, center] for center in graph.nodes) == 1, name=f"CriticalCareDemand_{node}")

        # Ensure routing to selected healthcare centers
        for node in graph.nodes:
            for center in graph.nodes:
                model.addCons(HealthcareRouting_vars[node, center] <= MedicalCenterSelection_vars[center], name=f"HealthcareServiceConstraint_{node}_{center}")

        # Service time constraints
        for node in graph.nodes:
            model.addCons(ServiceTime_vars[node] >= 0, name=f'ServiceMinTime_{node}')
            model.addCons(ServiceTime_vars[node] <= service_hours.max(), name=f'ServiceMaxTime_{node}')

        # Ensure at least 95% of facilities meet the delivery time window
        total_facilities = len(graph.nodes)
        min_satisfied = int(0.95 * total_facilities)
        model.addCons(quicksum(DeliveryWindowSatisfaction_vars[node] for node in graph.nodes) >= min_satisfied, name="MinDeliveryWindowSatisfaction")

        # Link service time with delivery window satisfaction variables
        for node in graph.nodes:
            model.addCons(ServiceTime_vars[node] <= DeliveryWindowSatisfaction_vars[node] * service_hours.max(), name=f"DeliveryWindowLink_{node}")

        # Restrict access to certain regions
        for node in graph.nodes:
            model.addCons(
                quicksum(HealthcareRouting_vars[node, center] for center in graph.nodes if access_restrictions[center] == 1) == 0,
                name=f"RestrictedAccess_{node}"
            )

        # Prioritize safety of women and children
        for node in graph.nodes:
            if priority_safety[node] == 1:
                model.addCons(
                    quicksum(MedicalCenterSelection_vars[neighbor] for neighbor in graph.neighbors[node] if security_risks[neighbor] <= 5.0) >= 1,
                    name=f"PrioritizeSafety_{node}"
                )

        # Minimize resource theft risk
        for i, j in graph.edges:
            if access_restrictions[i] == 0 and access_restrictions[j] == 0:
                model.addCons(
                    HealthcareRouting_vars[i, j] * theft_risks[i, j] <= ResourceTheftRisk_vars[i, j],
                    name=f"ResourceTheftRiskConstraint_{i}_{j}"
                )

        # Auction constraints: Ensure each bundle can be allocated to at most one center
        for i, bundle in enumerate(bundles):
            model.addCons(
                quicksum(MedicalCenterSelection_vars[center] for center in bundle) <= len(bundle) * ResourceAllocation_vars[i],
                name=f"BundleAllocation_{i}"
            )

        # Renewable constraints
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(
                    RenewableEnergy_vars[i, j] <= HealthcareRouting_vars[i, j],
                    name=f"RenewableEnergyLimit_{i}_{j}"
                )

        # Carbon emission constraint
        model.addCons(quicksum(carbon_emissions[i, j] * HealthcareRouting_vars[i, j] for i in graph.nodes for j in graph.nodes) <= self.carbon_limit, name="CarbonLimit")

        # Piecewise linearization constraints for response and energy times
        for i in graph.nodes:
            for j in graph.nodes:
                for k in range(segments):
                    model.addCons(ResponseTimePiecewise_vars[i, j, k] <= piecewise_slopes[k] * HealthcareRouting_vars[i, j], name=f"ResponseSegment_{i}_{j}_{k}")
                    model.addCons(EnergyPiecewise_vars[i,j,k] <= piecewise_slopes[k] * RenewableEnergy_vars[i,j], name=f"EnergySegment_{i}_{j}_{k}")
                    model.addCons(EmissionsPiecewise_vars[i,j,k] <= piecewise_slopes[k] * HealthcareRouting_vars[i,j], name=f"EmissionsSegment_{i}_{j}_{k}")

        # Objective: Minimize total costs and maximize auction revenue
        zoning_cost = quicksum(MedicalCenterSelection_vars[node] * zoning_costs[node] for node in graph.nodes)
        transportation_cost = quicksum(HealthcareRouting_vars[i, j] * transportation_costs[i, j] for i in graph.nodes for j in graph.nodes)
        theft_risk_cost = quicksum(ResourceTheftRisk_vars[i, j] for i in graph.nodes for j in graph.nodes)
        auction_revenue = quicksum(ResourceAllocation_vars[i] * prices[i] for i in range(len(bundles)))

        renewable_energy_cost = quicksum(renewable_energy_costs[i, j] * RenewableEnergy_vars[i, j] for i in graph.nodes for j in graph.nodes)
        hazardous_material_cost = quicksum(hazardous_material_penalty[node] * DeliveryWindowSatisfaction_vars[node] for node in graph.nodes)

        total_cost = zoning_cost + transportation_cost + theft_risk_cost + renewable_energy_cost + hazardous_material_cost - auction_revenue
        
        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 50,
        'edge_probability': 0.24,
        'graph_type': 'erdos_renyi',
        'edges_to_attach': 12,
        'n_bundles': 700,
        'carbon_limit': 3000,
    }

    humanitarian_optimization = HumanitarianAidDistribution(parameters, seed=seed)
    instance = humanitarian_optimization.get_instance()
    solve_status, solve_time, obj_val = humanitarian_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {obj_val:.2f}")