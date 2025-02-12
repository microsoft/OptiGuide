import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

############# Helper function #############
class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
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
        return Graph(number_of_nodes, edges, degrees, neighbors)

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        assert affinity >= 1 and affinity < number_of_nodes
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)
        return Graph(number_of_nodes, edges, degrees, neighbors)
############# Helper function #############

class EVStationPlacement:
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
            return Graph.erdos_renyi(self.n_nodes, self.edge_probability)
        elif self.graph_type == 'barabasi_albert':
            return Graph.barabasi_albert(self.n_nodes, self.affinity)
        else:
            raise ValueError("Unsupported graph type.")

    def generate_instance(self):
        graph = self.generate_graph()
        weights = {edge: random.randint(self.weight_low, self.weight_high) for edge in graph.edges}
        land_costs = np.random.randint(10, 100, self.n_nodes)
        zoning_compatibility = np.random.randint(0, 2, self.n_nodes)
        energy_availability = np.random.randint(50, 150, self.n_nodes)

        res = {'graph': graph, 'weights': weights, 'land_costs': land_costs,
               'zoning_compatibility': zoning_compatibility, 'energy_availability': energy_availability}
        
        capacity_penalties = np.random.randint(self.penalty_low, self.penalty_high, self.n_nodes)
        
        # New instance data for manufacturing context
        labor_costs = np.random.randint(50, 200, self.num_shifts)
        machine_speeds = np.random.randint(5, 20, self.n_machines)
        machine_break_downtime = np.random.randint(1, 5, self.n_machines)
        skilled_worker_availability = np.random.randint(0, 10, self.num_shifts)

        res.update({'capacity_penalties': capacity_penalties,
                    'labor_costs': labor_costs,
                    'machine_speeds': machine_speeds,
                    'machine_break_downtime': machine_break_downtime,
                    'skilled_worker_availability': skilled_worker_availability,
                    'shift_durations': self.shift_durations})
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        graph = instance['graph']
        weights = instance['weights']
        land_costs = instance['land_costs']
        zoning_compatibility = instance['zoning_compatibility']
        energy_availability = instance['energy_availability']
        capacity_penalties = instance['capacity_penalties']
        
        # New variables for manufacturing context
        labor_costs = instance['labor_costs']
        machine_speeds = instance['machine_speeds']
        machine_break_downtime = instance['machine_break_downtime']
        skilled_worker_availability = instance['skilled_worker_availability']
        shift_durations = instance['shift_durations']

        model = Model("EVStationPlacement")
        x, y, z, s, aux_c, aux_z = {}, {}, {}, {}, {}, {}
        
        # New variables for manufacturing
        m = {}  # machine operation
        shift_sched = {}  # shift scheduling

        for u in graph.nodes:
            x[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="x_%s" % u)
            z[u] = model.addVar(vtype='B', lb=0.0, ub=1, name="z_%s" % u)
            s[u] = model.addVar(vtype='C', lb=0.0, ub=self.max_oper_capacity, name="s_%s" % u)
            aux_c[u] = model.addVar(vtype='C', lb=0.0, ub=self.max_oper_capacity, name="aux_c_%s" % u)
            aux_z[u] = model.addVar(vtype='C', lb=0.0, ub=1, name="aux_z_%s" % u)

            model.addCons(s[u] >= self.min_oper_capacity * x[u], "MinOperCap_%s" % u)
            model.addCons(aux_c[u] == s[u] - x[u] * self.min_oper_capacity, "AuxCap_%s" % u)
            model.addCons(z[u] <= zoning_compatibility[u], "Zoning_Compat_%s" % u)
            model.addCons(aux_z[u] == z[u] * land_costs[u], "AuxLandCost_%s" % u)

        for e in graph.edges:
            y[e] = model.addVar(vtype='B', lb=0.0, ub=1, name="y_%s_%s" % (e[0], e[1]))
            model.addCons(y[e] <= x[e[0]] + x[e[1]], "C1_%s_%s" % (e[0], e[1]))
            model.addCons(y[e] <= 2 - x[e[0]] - x[e[1]], "C2_%s_%s" % (e[0], e[1]))

        for u in graph.nodes:
            model.addCons(x[u] * self.energy_demand <= energy_availability[u], "Energy_Lim_%s" % u)

        for machine in range(self.n_machines):
            m[machine] = model.addVar(vtype='C', lb=0.0, ub=1, name="m_%s" % machine)

        for shift in range(self.num_shifts):
            shift_sched[shift] = model.addVar(vtype='B', lb=0.0, ub=1, name="shift_%s" % shift)
            model.addCons(sum(m[machine] for machine in range(self.n_machines)) <= skilled_worker_availability[shift], 
                          "Skilled_Worker_Lim_%s" % shift)

        for machine in range(self.n_machines):
            for shift in range(self.num_shifts):
                model.addCons(m[machine] * machine_speeds[machine] <= shift_sched[shift] * shift_durations[shift], 
                              "Machine_Speed_Lim_%s_%s" % (machine, shift))
                model.addCons(m[machine] <= 1 - (machine_break_downtime[machine] / sum(shift_durations)),
                              "Machine_Breakdown_%s" % machine)

        # Objective with additional cost considerations
        objective_expr = quicksum(weights[e] * y[e] for e in graph.edges) - \
                         quicksum(aux_z[u] for u in graph.nodes) - \
                         quicksum(capacity_penalties[u] * aux_c[u] for u in graph.nodes) - \
                         quicksum(labor_costs[shift] * shift_sched[shift] for shift in range(self.num_shifts))

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42

    parameters = {
        'n_nodes': 450,
        'edge_probability': 0.52,
        'affinity': 2,
        'graph_type': 'barabasi_albert',
        'weight_low': 0,
        'weight_high': 2500,
        'energy_demand': 3,
        'min_oper_capacity': 12,
        'max_oper_capacity': 675,
        'penalty_low': 2,
        'penalty_high': 10,
        'num_shifts': 0,
        'n_machines': 3,
        'shift_durations': (4, 4, 4),
    }

    ev_station_placement = EVStationPlacement(parameters, seed=seed)
    instance = ev_station_placement.generate_instance()
    solve_status, solve_time = ev_station_placement.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")