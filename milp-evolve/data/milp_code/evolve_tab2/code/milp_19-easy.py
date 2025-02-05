import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum

# Include graph helper functions for generating resource or shift allocation graphs
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
                neighbor_prob = degrees[:new_node] / np.maximum((2 * len(edges)), 1)
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)
        return Graph(number_of_nodes, edges, degrees, neighbors)

class MultiItemLotSizing:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_resource_constraints(self):
        machine_capacities = np.random.randint(50, 150, size=self.num_resources)
        labor_hours = np.random.randint(30, 100, size=self.num_resources)
        return machine_capacities, labor_hours

    def generate_instance(self):
        setup_costs, setup_times, variable_costs, demands, holding_costs, resource_upper_bounds = {}, {}, {}, {}, {}, {}

        sumT = 0
        for t in range(1, self.num_periods+1):
            for p in range(1, self.num_products+1):
                setup_times[t, p] = 10 * random.randint(1, 5)  
                setup_costs[t, p] = 100 * random.randint(1, 10) 
                variable_costs[t, p] = 0         

                demands[t, p] = 100 + random.randint(-25, 25) 
                if t <= 4:
                    if random.random() < 0.25:     
                        demands[t,p] = 0
                sumT += setup_times[t, p] + demands[t, p]
                holding_costs[t, p] = random.randint(1, 5)

        for t in range(1, self.num_periods+1):
            resource_upper_bounds[t] = int(float(sumT) / (float(self.num_periods) * self.factor))

        machine_capacities, labor_hours = self.generate_resource_constraints()

        res = {
            'setup_costs': setup_costs,
            'setup_times': setup_times,
            'variable_costs': variable_costs,
            'demands': demands,
            'holding_costs': holding_costs,
            'resource_upper_bounds': resource_upper_bounds,
            'machine_capacities': machine_capacities,
            'labor_hours': labor_hours
        }
        return res

    def solve(self, instance):

        setup_costs = instance['setup_costs']
        setup_times = instance['setup_times']
        variable_costs = instance['variable_costs']
        demands = instance['demands']
        holding_costs = instance['holding_costs']
        resource_upper_bounds = instance['resource_upper_bounds']
        machine_capacities = instance['machine_capacities']
        labor_hours = instance['labor_hours']

        model = Model("enhanced multi-item lotsizing")

        y, x, I, shift = {}, {}, {}, {}
        for p in range(1, self.num_products+1):
            for t in range(1, self.num_periods+1):
                y[t, p] = model.addVar(vtype="B", name="y_%s_%s" % (t,p))
                x[t, p] = model.addVar(vtype="C", name="x_%s_%s" % (t,p))
                I[t, p] = model.addVar(vtype="C", name="I_%s_%s" % (t,p))
            I[0, p] = 0

        for t in range(1, self.num_periods+1):
            # time capacity constraints
            model.addCons(quicksum(setup_times[t, p] * y[t, p] + x[t, p] 
                                   for p in range(1, self.num_products+1)) <= resource_upper_bounds[t], "TimeUB_%s" % t)

            for p in range(1, self.num_products+1):
                # flow conservation constraints
                model.addCons(I[t-1, p] + x[t,p] == I[t, p] + demands[t,p], "FlowCons_%s_%s" % (t, p))

                # capacity connection constraints
                model.addCons(x[t,p] <= (resource_upper_bounds[t] - setup_times[t,p]) * y[t, p], "ConstrUB_%s_%s" % (t, p))

                # tighten constraints
                model.addCons(x[t,p] <= demands[t,p] * y[t,p] + I[t,p], "Tighten_%s_%s" % (t, p))

        # Add constraints from the second MILP
        for r in range(self.num_resources):
            for t in range(1, self.num_periods+1):
                shift[t, r] = model.addVar(vtype="B", name="shift_%s_%s" % (t, r))
                model.addCons(quicksum(setup_times[t, p] * shift[t, r] for p in range(1, self.num_products+1)) 
                              <= machine_capacities[r], "MachineCap_%s_%s" % (t, r))
                model.addCons(quicksum(x[t, p] * shift[t, r] for p in range(1, self.num_products+1)) 
                              <= labor_hours[r], "LaborHours_%s_%s" % (t, r))

        # Modify objective to consider a combination of both costs and resource utilization
        objective_expr = quicksum(setup_costs[t,p] * y[t,p] + 
                                  variable_costs[t,p] * x[t,p] + 
                                  holding_costs[t,p] * I[t,p] 
                                  for t in range(1, self.num_periods+1) for p in range(1, self.num_products+1)) \
                        - quicksum(shift[t, r] for t in range(1, self.num_periods+1) for r in range(self.num_resources))
        
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == "__main__":
    parameters = {
        'num_periods': 30,
        'num_products': 10,
        'factor': 1.0,
        'num_resources': 5  # New parameter for number of additional resources (shifts, machines)
    }

    model = MultiItemLotSizing(parameters)
    instance = model.generate_instance()
    solve_status, solve_time = model.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")