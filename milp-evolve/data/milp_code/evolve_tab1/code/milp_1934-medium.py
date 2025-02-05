import random
import time
import numpy as np
import networkx as nx
from itertools import permutations
from pyscipopt import Model, quicksum

class Graph:
    """Helper function: Container for a graph."""
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """Generate an Erdös-Rényi random graph with a given edge probability."""
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in permutations(np.arange(number_of_nodes), 2):
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
        """Generate a Barabási-Albert random graph."""
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

    @staticmethod
    def watts_strogatz(number_of_nodes, k, p):
        """Generate a Watts-Strogatz small-world graph."""
        edges = set()
        neighbors = {node: set() for node in range(number_of_nodes)}
        G = nx.watts_strogatz_graph(number_of_nodes, k, p)
        degrees = np.zeros(number_of_nodes, dtype=int)
        for edge in G.edges:
            edges.add(edge)
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

class EnergyInvestment:
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
        elif self.graph_type == 'watts_strogatz':
            return Graph.watts_strogatz(self.n_nodes, self.k, self.rewiring_prob)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        energy_demands = np.random.randint(50, 150, size=graph.number_of_nodes)  # Increased demand for real-world context
        investment_costs = np.random.randint(500, 1000, size=graph.number_of_nodes)  # Costs for investment venues
        operation_costs = np.random.randint(1, 100, size=(graph.number_of_nodes, graph.number_of_nodes))

        # Investment parameters
        installation_costs = np.random.randint(100, 300, size=graph.number_of_nodes)
        distribution_costs = np.random.rand(graph.number_of_nodes, graph.number_of_nodes) * 50
        
        max_budget = np.random.randint(3000, 10000)  # Higher budget for larger investment projects
        min_venues = 3
        max_venues = 15
        generation_capacities = np.random.randint(100, 1000, size=graph.number_of_nodes)
        unmet_penalties = np.random.randint(20, 100, size=graph.number_of_nodes)
        environmental_impact = np.random.randint(200, 1000, size=graph.number_of_nodes)

        # Additional parameters for new constraints
        available_funds = np.random.randint(50, 300, size=graph.number_of_nodes)
        fund_costs = np.random.rand(graph.number_of_nodes) * 10
        zero_carbon_penalties = np.random.randint(10, 100, size=graph.number_of_nodes)

        # New data for Big M constraints
        max_generation_times = np.random.randint(30, 120, size=graph.number_of_nodes)
        BigM = 1e6  # Large constant for Big M formulation
        
        # Infrastructure limits for generation
        infrastructure_limits = np.random.randint(1000, 5000, size=graph.number_of_nodes)

        # Environmental Impact
        environmental_needs = np.random.uniform(0, 1, size=graph.number_of_nodes)

        # Logical condition data for edge existence
        edge_exists = {(i, j): (1 if (i, j) in graph.edges else 0) for i in graph.nodes for j in graph.nodes}

        res = {
            'graph': graph,
            'energy_demands': energy_demands,
            'investment_costs': investment_costs,
            'operation_costs': operation_costs,
            'installation_costs': installation_costs,
            'distribution_costs': distribution_costs,
            'max_budget': max_budget,
            'min_venues': min_venues,
            'max_venues': max_venues,
            'generation_capacities': generation_capacities,
            'unmet_penalties': unmet_penalties,
            'environmental_impact': environmental_impact,
            'available_funds': available_funds,
            'fund_costs': fund_costs,
            'zero_carbon_penalties': zero_carbon_penalties,
            'max_generation_times': max_generation_times,
            'BigM': BigM,
            'infrastructure_limits': infrastructure_limits,
            'environmental_needs': environmental_needs,
            'edge_exists': edge_exists  # Added edge existence data
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        graph = instance['graph']
        energy_demands = instance['energy_demands']
        investment_costs = instance['investment_costs']
        installation_costs = instance['installation_costs']
        distribution_costs = instance['distribution_costs']
        max_budget = instance['max_budget']
        min_venues = instance['min_venues']
        max_venues = instance['max_venues']
        generation_capacities = instance['generation_capacities']
        unmet_penalties = instance['unmet_penalties']
        environmental_impact = instance['environmental_impact']
        available_funds = instance['available_funds']
        fund_costs = instance['fund_costs']
        zero_carbon_penalties = instance['zero_carbon_penalties']
        max_generation_times = instance['max_generation_times']
        BigM = instance['BigM']
        infrastructure_limits = instance['infrastructure_limits']
        environmental_needs = instance['environmental_needs']
        edge_exists = instance['edge_exists']  # Retrieved edge existence data

        model = Model("EnergyInvestment")

        # Add variables
        venue_vars = {node: model.addVar(vtype="B", name=f"VenueSelection_{node}") for node in graph.nodes}
        allocation_vars = {(i, j): model.addVar(vtype="B", name=f"Allocation_{i}_{j}") for i in graph.nodes for j in graph.nodes}
        penalty_vars = {node: model.addVar(vtype="C", name=f"Penalty_{node}") for node in graph.nodes}

        # New variables for investment capacity, environmental impact, and zero carbon footprints
        capacity_vars = {node: model.addVar(vtype="C", name=f"GenerationCapacity_{node}") for node in graph.nodes}
        impact_vars = {node: model.addVar(vtype="C", name=f"Impact_{node}") for node in graph.nodes}
        zero_carbon_vars = {node: model.addVar(vtype="C", name=f"CarbonFree_{node}") for node in graph.nodes}

        # New variables for generation times
        generation_time_vars = {node: model.addVar(vtype="C", name=f"GenerationTime_{node}") for node in graph.nodes}

        # New variables for environmental needs
        fund_vars = {node: model.addVar(vtype="C", name=f"Fund_{node}") for node in graph.nodes}

        # Number of investment venues constraint
        model.addCons(quicksum(venue_vars[node] for node in graph.nodes) >= min_venues, name="MinVenues")
        model.addCons(quicksum(venue_vars[node] for node in graph.nodes) <= max_venues, name="MaxVenues")

        # Demand satisfaction constraints with penalties
        for zone in graph.nodes:
            model.addCons(
                quicksum(allocation_vars[zone, center] for center in graph.nodes) + penalty_vars[zone] == 1, 
                name=f"Demand_{zone}"
            )

        # Allocation from open venues with logical condition for edge existence
        for i in graph.nodes:
            for j in graph.nodes:
                model.addCons(allocation_vars[i, j] <= venue_vars[j], name=f"AllocationService_{i}_{j}")
                model.addCons(allocation_vars[i, j] <= edge_exists[i, j], name=f"AllocationEdgeExists_{i}_{j}")  # Logical condition

        # Capacity constraints with logical condition for fund availability
        for j in graph.nodes:
            model.addCons(quicksum(allocation_vars[i, j] * energy_demands[i] for i in graph.nodes) <= generation_capacities[j], name=f"Capacity_{j}")
            model.addCons(capacity_vars[j] <= BigM * quicksum(allocation_vars[i, j] for i in graph.nodes), name=f"CapacityLogic_{j}")  # Logical condition

        # Budget constraints
        total_cost = quicksum(venue_vars[node] * installation_costs[node] for node in graph.nodes) + \
                     quicksum(allocation_vars[i, j] * distribution_costs[i, j] for i in graph.nodes for j in graph.nodes) + \
                     quicksum(penalty_vars[node] * unmet_penalties[node] for node in graph.nodes)

        model.addCons(total_cost <= max_budget, name="Budget")

        # New resource constraints
        for node in graph.nodes:
            model.addCons(capacity_vars[node] <= available_funds[node], name=f"Capacity_{node}")

        # New environmental impact constraints
        total_impact = quicksum(impact_vars[node] * environmental_impact[node] for node in graph.nodes)
        model.addCons(total_impact >= self.min_environmental_impact, name="Impact")

        # New zero carbon constraints
        total_carbon = quicksum(zero_carbon_vars[node] * zero_carbon_penalties[node] for node in graph.nodes)
        model.addCons(total_carbon <= self.zero_carbon_threshold, name="CarbonFree")

        # Generation time limits using Big M formulation
        for node in graph.nodes:
            model.addCons(generation_time_vars[node] <= max_generation_times[node], name=f"MaxGenerationTime_{node}")
            model.addCons(generation_time_vars[node] <= BigM * allocation_vars[node, node], name=f"BigMGenerationTime_{node}")

        # If allocation is used, venue must be open
        for node in graph.nodes:
            model.addCons(capacity_vars[node] <= BigM * venue_vars[node], name=f"AllocationVenue_{node}")

        # Environmental needs constraints
        for node in graph.nodes:
            model.addCons(fund_vars[node] >= environmental_needs[node], name=f"Fund_{node}")
        
        # Ensure supply does not exceed local infrastructure limits
        for node in graph.nodes:
            model.addCons(fund_vars[node] <= infrastructure_limits[node], name=f"Infrastructure_{node}")

        # New objective: Minimize total investment cost including cost and penalties
        objective = total_cost + quicksum(impact_vars[node] for node in graph.nodes) + \
                    quicksum(zero_carbon_vars[node] * zero_carbon_penalties[node] for node in graph.nodes)

        model.setObjective(objective, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 75,
        'edge_probability': 0.73,
        'graph_type': 'erdos_renyi',
        'k': 80,
        'rewiring_prob': 0.73,
        'max_capacity': 600,
        'min_environmental_impact': 6000,
        'zero_carbon_threshold': 5000,
        'BigM': 1000000.0,
    }

    energy_investment = EnergyInvestment(parameters, seed=seed)
    instance = energy_investment.generate_instance()
    solve_status, solve_time = energy_investment.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")