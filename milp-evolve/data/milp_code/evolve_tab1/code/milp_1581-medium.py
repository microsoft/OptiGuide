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
        population = np.random.normal(loc=self.population_mean, scale=self.population_std, size=self.number_of_neighborhoods).astype(int)
        population = np.clip(population, self.min_population, self.max_population)

        # Generating the recycling rates
        recycling_rates = np.random.uniform(self.min_recycling_rate, self.max_recycling_rate, size=self.number_of_neighborhoods)
        
        # Distance matrix between neighborhoods and potential recycling center locations
        distances = np.random.uniform(self.min_distance, self.max_distance, (self.number_of_neighborhoods, self.potential_locations))

        # Generating facility costs and capacities
        facility_costs = np.random.uniform(self.min_facility_cost, self.max_facility_cost, size=self.potential_locations)
        facility_capacities = np.random.uniform(self.min_facility_capacity, self.max_facility_capacity, size=self.potential_locations).astype(int)
        
        # Sustainable energy availability
        energy_availability = np.random.uniform(self.min_energy_availability, self.max_energy_availability, size=self.potential_locations)

        res = {
            'population': population, 
            'recycling_rates': recycling_rates,
            'distances': distances, 
            'facility_costs': facility_costs, 
            'facility_capacities': facility_capacities,
            'energy_availability': energy_availability
        }

        return res        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        population = instance['population']
        recycling_rates = instance['recycling_rates']
        distances = instance['distances']
        facility_costs = instance['facility_costs']
        facility_capacities = instance['facility_capacities']
        energy_availability = instance['energy_availability']
        
        neighborhoods = range(len(population))
        locations = range(len(facility_costs))
        
        model = Model("RecyclingOptimization")
        x = {}  # Binary variable: 1 if recycling center is placed at location l
        y = {}  # Binary variable: 1 if a collection route from n to l exists

        # Decision variables
        for l in locations:
            x[l] = model.addVar(vtype="B", name=f"x_{l}")
            for n in neighborhoods:
                y[n, l] = model.addVar(vtype="B", name=f"y_{n}_{l}")

        # Objective: Minimize transportation cost + facility costs + environmental impact
        obj_expr = quicksum(distances[n][l] * y[n, l] * population[n] * recycling_rates[n] for n in neighborhoods for l in locations) + \
                   quicksum(facility_costs[l] * x[l] for l in locations)
        
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
                quicksum(y[n, l] * population[n] * recycling_rates[n] for n in neighborhoods) <= facility_capacities[l] * x[l],
                f"Capacity_{l}"
            )

        # Constraints: Sustainable energy availability at each facility
        for l in locations:
            model.addCons(
                quicksum(y[n, l] * recycling_rates[n] for n in neighborhoods) <= energy_availability[l],
                f"Energy_{l}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_neighborhoods': 150,
        'potential_locations': 100,
        'population_mean': 1500,
        'population_std': 2500,
        'min_population': 500,
        'max_population': 3000,
        'min_recycling_rate': 0.59,
        'max_recycling_rate': 0.62,
        'min_distance': 7,
        'max_distance': 40,
        'min_facility_cost': 3000,
        'max_facility_cost': 10000,
        'min_facility_capacity': 10000,
        'max_facility_capacity': 50000,
        'min_energy_availability': 5000,
        'max_energy_availability': 25000,
    }

    recycling_optimization = RecyclingOptimization(parameters, seed=seed)
    instance = recycling_optimization.generate_instance()
    solve_status, solve_time = recycling_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")