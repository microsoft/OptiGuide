import numpy as np
import random
import time
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
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

    def efficient_greedy_clique_partition(self):
        """
        Partition the graph into cliques using an efficient greedy algorithm.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

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
    def barabasi_albert(number_of_nodes, affinity):
        """
        Generate a Barabási-Albert random graph with a given edge probability.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2 * len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph
############# Helper function #############

class IndependentSet:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_graph(self):
        if self.graph_type == 'erdos_renyi':
            graph = Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            graph = Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")
        return graph

    def generate_instance(self):
        graph = self.generate_graph()
        cliques = graph.efficient_greedy_clique_partition()
        inequalities = set(graph.edges)
        for clique in cliques:
            clique = tuple(sorted(clique))
            for edge in combinations(clique, 2):
                inequalities.remove(edge)
            if len(clique) > 1:
                inequalities.add(clique)

        # Put trivial inequalities for nodes that didn't appear
        # in the constraints, otherwise SCIP will complain
        used_nodes = set()
        for group in inequalities:
            used_nodes.update(group)
        for node in range(10):
            if node not in used_nodes:
                inequalities.add((node,))

        # Define edge capacities
        edge_capacities = {edge: np.random.randint(1, 10) for edge in graph.edges}

        res = {'graph': graph, 'inequalities': inequalities, 'edge_capacities': edge_capacities}

        # Generate parking data
        parking_capacity = np.random.randint(1, self.max_parking_capacity, size=self.n_parking_zones)
        parking_zones = {i: np.random.choice(range(self.n_nodes), size=self.n_parking_in_zone, replace=False) for i in
                         range(self.n_parking_zones)}

        # Generate delivery time windows with uncertainties
        time_windows = {i: (np.random.randint(0, self.latest_delivery_time // 2),
                            np.random.randint(self.latest_delivery_time // 2, self.latest_delivery_time)) for i in range(self.n_nodes)}
        travel_times = {(u, v): np.random.randint(1, self.max_travel_time) for u, v in graph.edges}
        uncertainty = {i: np.random.normal(0, self.time_uncertainty_stddev, size=2) for i in range(self.n_nodes)}

        res.update({'parking_capacity': parking_capacity,
                    'parking_zones': parking_zones,
                    'time_windows': time_windows,
                    'travel_times': travel_times,
                    'uncertainty': uncertainty})
        
        # Generate autonomous vehicle data
        autonomous_vehicle_routes = {i: np.random.choice(range(self.n_nodes), size=self.av_route_length, replace=False) for i in range(self.n_autonomous_vehicles)}
        road_closures = np.random.choice(range(self.n_nodes), size=self.n_road_closures, replace=False)
        special_events = np.random.choice(range(self.n_nodes), size=self.n_special_events, replace=False)
        
        res.update({'autonomous_vehicle_routes': autonomous_vehicle_routes,
                    'road_closures': road_closures,
                    'special_events': special_events})

        # Generate adaptive traffic signal data
        adaptive_signals = {i: np.random.choice([True, False]) for i in range(self.n_nodes)}

        res.update({'adaptive_signals': adaptive_signals})

        # Generate eco-friendly options data
        eco_friendly_zones = np.random.choice(range(self.n_nodes), size=self.n_eco_friendly_zones, replace=False)
        co2_saving = {i: np.random.uniform(0, self.max_co2_saving) for i in eco_friendly_zones}

        res.update({'eco_friendly_zones': eco_friendly_zones,
                    'co2_saving': co2_saving})

        # New constraints and variables for sustainability
        res.update({'sustainability_constraint': np.random.uniform(0, self.min_sustainability_requirement)})
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        inequalities = instance['inequalities']
        edge_capacities = instance['edge_capacities']
        parking_capacity = instance['parking_capacity']
        parking_zones = instance['parking_zones']
        time_windows = instance['time_windows']
        travel_times = instance['travel_times']
        uncertainty = instance['uncertainty']
        autonomous_vehicle_routes = instance['autonomous_vehicle_routes']
        road_closures = instance['road_closures']
        special_events = instance['special_events']
        adaptive_signals = instance['adaptive_signals']
        eco_friendly_zones = instance['eco_friendly_zones']
        co2_saving = instance['co2_saving']
        sustainability_constraint = instance['sustainability_constraint']

        model = Model("IndependentSetWithLogistics")
        var_names = {}

        for node in graph.nodes:
            var_names[node] = model.addVar(vtype="B", name=f"x_{node}")

        for count, group in enumerate(inequalities):
            model.addCons(quicksum(var_names[node] for node in group) <= 1, name=f"clique_{count}")

        # Add edge capacity constraints
        flow_vars = {}
        for edge in graph.edges:
            u, v = edge
            flow_vars[edge] = model.addVar(vtype="C", name=f"flow_{u}_{v}")

        for edge in graph.edges:
            u, v = edge
            model.addCons(flow_vars[edge] <= edge_capacities[edge], name=f"capacity_{u}_{v}")

        for node in graph.nodes:
            model.addCons(quicksum(flow_vars[(u, v)] for v in graph.neighbors[node] if (u, v) in flow_vars) <= 1, name=f"flow_conservation_{node}")

        # Modified Constraints: Parking space constraints with Big M formulation
        M = self.big_M_constant
        parking_vars = {}
        for zone, cols in parking_zones.items():
            for col in cols:
                parking_vars[col] = model.addVar(vtype="B", name=f"p_{col}")
                model.addCons(var_names[col] <= parking_vars[col] * M, f"occupy_{col}_big_m")
                model.addCons(parking_vars[col] <= var_names[col] + (1 - var_names[col]) * M, f"occupy_{col}_reverse_big_m")

            # Constraint to ensure the number of occupied parking spaces in a zone is limited
            model.addCons(quicksum(parking_vars[col] for col in cols) <= parking_capacity[zone], f"parking_limit_{zone}")

        # Add variables and constraints for delivery time windows with uncertainty
        time_vars = {}
        early_penalty_vars = {}
        late_penalty_vars = {}
        for j in graph.nodes:
            time_vars[j] = model.addVar(vtype='C', name=f"t_{j}")
            early_penalty_vars[j] = model.addVar(vtype='C', name=f"e_{j}")
            late_penalty_vars[j] = model.addVar(vtype='C', name=f"l_{j}")
            
            # Incorporate the uncertainty in the delivery times
            start_window, end_window = time_windows[j]
            uncertainty_start, uncertainty_end = uncertainty[j]
            model.addCons(time_vars[j] >= start_window + uncertainty_start, f"time_window_start_{j}")
            model.addCons(time_vars[j] <= end_window + uncertainty_end, f"time_window_end_{j}")
            
            model.addCons(early_penalty_vars[j] >= start_window + uncertainty_start - time_vars[j], f"early_penalty_{j}")
            model.addCons(late_penalty_vars[j] >= time_vars[j] - (end_window + uncertainty_end), f"late_penalty_{j}")

        # Composite objective: Minimize total cost and maximize total flow
        flow_term = quicksum(flow_vars[(u, v)] for u, v in flow_vars)
        parking_penalty_term = quicksum(parking_vars[col] for col in parking_vars)
        time_penalty_term = quicksum(early_penalty_vars[j] + late_penalty_vars[j] for j in graph.nodes)

        # Add new constraints and objective terms for autonomous vehicle routes
        av_penalty_term = quicksum(quicksum(var_names[node] for node in route if node in road_closures or node in special_events) for route in autonomous_vehicle_routes.values())

        # Ensure route continuity without intersections with road closures or special events
        for route in autonomous_vehicle_routes.values():
            for node in route:
                for closure in road_closures:
                    model.addCons(var_names[node] + var_names[closure] <= 1, f"no_intersect_{node}_{closure}")
                for event in special_events:
                    model.addCons(var_names[node] + var_names[event] <= 1, f"no_intersect_{node}_{event}")

        # Ensure sustainability constraints are met
        model.addCons(quicksum(co2_saving[node] * var_names[node] for node in eco_friendly_zones) >= sustainability_constraint, "sustainability")

        # Objective function
        objective_expr = quicksum(var_names[node] for node in graph.nodes) + flow_term - self.parking_penalty_weight * parking_penalty_term - self.time_penalty_weight * time_penalty_term + av_penalty_term

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_nodes': 2250,
        'edge_probability': 0.46,
        'affinity': 4,
        'graph_type': 'barabasi_albert',
        'big_M_constant': 1000,
        'n_parking_zones': 100,
        'n_parking_in_zone': 200,
        'max_parking_capacity': 3000,
        'parking_penalty_weight': 0.66,
        'latest_delivery_time': 2880,
        'max_travel_time': 3000,
        'time_penalty_weight': 0.6,
        'time_uncertainty_stddev': 40,
        'n_autonomous_vehicles': 90,
        'av_route_length': 120,
        'n_road_closures': 175,
        'n_special_events': 2,
        'n_eco_friendly_zones': 250,
        'max_co2_saving': 2000,
        'min_sustainability_requirement': 500,
    }

    independent_set_problem = IndependentSet(parameters, seed=seed)
    instance = independent_set_problem.generate_instance()
    solve_status, solve_time = independent_set_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")