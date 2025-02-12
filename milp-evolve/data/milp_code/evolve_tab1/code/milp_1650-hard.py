import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class RecyclingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        neighborhoods = range(self.number_of_neighborhoods)
        locations = range(self.potential_locations)
        
        # Generating neighborhood population
        population = np.random.gamma(shape=self.population_shape, scale=self.population_scale, size=self.number_of_neighborhoods).astype(int)
        population = np.clip(population, self.min_population, self.max_population)

        # Generating the recycling rates
        recycling_rates = np.random.beta(self.alpha_recycling_rate, self.beta_recycling_rate, size=self.number_of_neighborhoods)
        
        # Distance matrix between neighborhoods and potential recycling center locations
        G = nx.barabasi_albert_graph(self.number_of_neighborhoods, self.number_of_graph_edges)
        distances = nx.floyd_warshall_numpy(G) * np.random.uniform(self.min_distance_multiplier, self.max_distance_multiplier)

        # Generating facility costs and capacities
        facility_costs = np.random.gamma(self.facility_cost_shape, self.facility_cost_scale, size=self.potential_locations)
        facility_capacities = np.random.gamma(self.facility_capacity_shape, self.facility_capacity_scale, size=self.potential_locations).astype(int)

        res = {
            'population': population, 
            'recycling_rates': recycling_rates,
            'distances': distances, 
            'facility_costs': facility_costs, 
            'facility_capacities': facility_capacities
        }
        
        return res        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        population = instance['population']
        recycling_rates = instance['recycling_rates']
        distances = instance['distances']
        facility_costs = instance['facility_costs']
        facility_capacities = instance['facility_capacities']
        
        neighborhoods = range(len(population))
        locations = range(len(facility_costs))
        
        model = Model("RecyclingOptimization")
        x = {}  # Binary variable: 1 if recycling center is placed at location l
        y = {}  # Binary variable: 1 if a collection route from n to l exists
        d = {}  # Continuous variable: Fraction of demand satisfied
        
        # Decision variables
        for l in locations:
            x[l] = model.addVar(vtype="B", name=f"x_{l}")
            for n in neighborhoods:
                y[n, l] = model.addVar(vtype="B", name=f"y_{n}_{l}")
                d[n, l] = model.addVar(vtype="C", name=f"d_{n}_{l}", lb=0, ub=1)
        
        # Objective: Minimize transportation cost + facility costs + environmental impact + maximize satisfaction
        obj_expr = quicksum(distances[n][l] * y[n, l] * population[n] * recycling_rates[n] for n in neighborhoods for l in locations) + \
                   quicksum(facility_costs[l] * x[l] for l in locations) + \
                   quicksum((1 - d[n, l]) * population[n] for n in neighborhoods for l in locations) - \
                   quicksum(d[n, l] for n in neighborhoods for l in locations)
        
        model.setObjective(obj_expr, "minimize")

        # Constraints: Each neighborhood must have access to one recycling facility
        for n in neighborhoods:
            model.addCons(
                quicksum(y[n, l] for l in locations) == 1,
                f"Access_{n}"
            )

        # Constraints: Facility capacity must not be exceeded
        for l in locations:
            model.addCons(
                quicksum(y[n, l] * population[n] * recycling_rates[n] * d[n, l] for n in neighborhoods) <= facility_capacities[l] * x[l],
                f"Capacity_{l}"
            )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_neighborhoods': 75,
        'potential_locations': 50,
        'population_shape': 25,
        'population_scale': 1500,
        'min_population': 2500,
        'max_population': 3000,
        'alpha_recycling_rate': 48.0,
        'beta_recycling_rate': 33.75,
        'number_of_graph_edges': 30,
        'min_distance_multiplier': 7,
        'max_distance_multiplier': 144,
        'facility_cost_shape': 15.0,
        'facility_cost_scale': 2000.0,
        'facility_capacity_shape': 400.0,
        'facility_capacity_scale': 2000.0,
    }
    
    recycling_optimization = RecyclingOptimization(parameters, seed=seed)
    instance = recycling_optimization.generate_instance()
    solve_status, solve_time = recycling_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")