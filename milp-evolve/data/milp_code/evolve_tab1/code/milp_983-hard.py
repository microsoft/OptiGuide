import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MilkCollectionNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_graph(self):
        n_cities = np.random.randint(self.min_n, self.max_n)
        G = nx.barabasi_albert_graph(n=n_cities, m=self.ba_m, seed=self.seed)
        return G

    def generate_milk_cities(self, G):
        for node in G.nodes:
            G.nodes[node]['milk'] = np.random.randint(100, 1000)

    def generate_possible_stops(self, G):
        edges = list(G.edges)
        random.shuffle(edges)
        nbr_stops = min(self.max_possible_stops, len(edges))
        Stops = set(edges[:nbr_stops])
        return Stops

    def generate_transport_revenues(self, G, Stops):
        transport_revenues = {}
        for u, v in Stops:
            transport_revenues[(u, v)] = np.random.randint(300, 1000)
        return transport_revenues

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_milk_cities(G)
        Stops = self.generate_possible_stops(G)
        transport_revenues = self.generate_transport_revenues(G, Stops)
        
        res = {'G': G, 
               'Stops': Stops,
               'transport_revenues': transport_revenues}
        
        res['max_stops'] = np.random.randint(self.min_possible_stops, self.max_possible_stops)
        res['min_visited_cities'] = self.min_visited_cities
        res['max_visited_cities'] = self.max_visited_cities
        
        ### given instance data code ends here
        ### new instance data code ends here
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G, Stops, transport_revenues = instance['G'], instance['Stops'], instance['transport_revenues']
        max_stops = instance['max_stops']
        min_visited_cities = instance['min_visited_cities']
        max_visited_cities = instance['max_visited_cities']

        model = Model("MilkCollectionNetwork")

        # Define Variables
        milk_vars = {f"milk{node}": model.addVar(vtype="B", name=f"milk{node}") for node in G.nodes}
        stop_vars = {f"stop{u}_{v}": model.addVar(vtype="B", name=f"stop{u}_{v}") for u, v in Stops}

        # Objective: Maximize milk collection with transport revenues
        objective_expr = quicksum(G.nodes[node]['milk'] * milk_vars[f"milk{node}"] for node in G.nodes) + \
                         quicksum(transport_revenues[u, v] * stop_vars[f"stop{u}_{v}"] for u, v in Stops)

        # Constraints
        for u, v in G.edges:
            if (u, v) in Stops:
                model.addCons(milk_vars[f"milk{u}"] + milk_vars[f"milk{v}"] - stop_vars[f"stop{u}_{v}"] <= 1, name=f"NewFrontier_{u}_{v}")
            else:
                model.addCons(milk_vars[f"milk{u}"] + milk_vars[f"milk{v}"] <= 1, name=f"CityConstraints_{u}_{v}")

        # Additional stop constraints
        model.addCons(quicksum(stop_vars[f"stop{u}_{v}"] for u, v in Stops) <= max_stops, name="Max_Stops")
        
        # Bounds on visited cities
        model.addCons(quicksum(milk_vars[f"milk{node}"] for node in G.nodes) >= min_visited_cities, name="Min_Visited_Cities")
        model.addCons(quicksum(milk_vars[f"milk{node}"] for node in G.nodes) <= max_visited_cities, name="Max_Visited_Cities")

        ### given constraints and variables and objective code ends here
        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 150,
        'max_n': 187,
        'set_type': 'NEWSET',
        'set_param': 3500.0,
        'alpha': 0.31,
        'ba_m': 28,
        'min_possible_stops': 300,
        'max_possible_stops': 1200,
        'min_visited_cities': 20,
        'max_visited_cities': 1050,
    }
    
    ### given parameter code ends here
    ### new parameter code ends here

    mc_network = MilkCollectionNetwork(parameters, seed=seed)
    instance = mc_network.generate_instance()
    solve_status, solve_time = mc_network.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")