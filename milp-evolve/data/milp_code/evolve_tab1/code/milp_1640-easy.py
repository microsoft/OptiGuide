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

        # Sustainable energy availability
        energy_availability = np.random.weibull(self.energy_shape, size=self.potential_locations) * self.energy_scale
        
        # Breakpoints for piecewise linear functions
        d_breakpoints = np.linspace(0, 1, self.num_segments + 1) 
        transport_costs_segments = np.random.uniform(self.min_transport_cost, self.max_transport_cost, self.num_segments)
        
        res = {
            'population': population, 
            'recycling_rates': recycling_rates,
            'distances': distances, 
            'facility_costs': facility_costs, 
            'facility_capacities': facility_capacities,
            'energy_availability': energy_availability,
            'd_breakpoints': d_breakpoints,
            'transport_costs_segments': transport_costs_segments
        }
        
        # New instance data for set covering formulation
        min_service_centers = np.random.randint(1, self.min_centers, size=self.number_of_neighborhoods)
        res['min_service_centers'] = min_service_centers
        
        return res        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        population = instance['population']
        recycling_rates = instance['recycling_rates']
        distances = instance['distances']
        facility_costs = instance['facility_costs']
        facility_capacities = instance['facility_capacities']
        energy_availability = instance['energy_availability']
        d_breakpoints = instance['d_breakpoints']
        transport_costs_segments = instance['transport_costs_segments']
        min_service_centers = instance['min_service_centers']
        
        neighborhoods = range(len(population))
        locations = range(len(facility_costs))
        segments = range(len(transport_costs_segments))
        
        model = Model("RecyclingOptimization")
        x = {}  # Binary variable: 1 if recycling center is placed at location l
        y = {}  # Binary variable: 1 if a collection route from n to l exists
        f = {}  # Integer variable: Collection frequency
        d = {}  # Continuous variable: Fraction of demand satisfied
        z_nl_s = {}  # Binary variable for piecewise linear segment
        cover = {}  # Auxiliary binary variables for set covering
        
        # Decision variables
        for l in locations:
            x[l] = model.addVar(vtype="B", name=f"x_{l}")
            for n in neighborhoods:
                y[n, l] = model.addVar(vtype="B", name=f"y_{n}_{l}")
                f[n, l] = model.addVar(vtype="I", name=f"f_{n}_{l}", lb=0, ub=self.max_collection_frequency)
                d[n, l] = model.addVar(vtype="C", name=f"d_{n}_{l}", lb=0, ub=1)
                for s in segments:
                    z_nl_s[n, l, s] = model.addVar(vtype="B", name=f"z_{n}_{l}_{s}")
            cover[l] = model.addVar(vtype="B", name=f"cover_{l}")
        
        # Objective: Minimize transportation cost + facility costs + environmental impact + maximize satisfaction
        obj_expr = quicksum(distances[n][l] * y[n, l] * population[n] * recycling_rates[n] for n in neighborhoods for l in locations) + \
                   quicksum(facility_costs[l] * x[l] for l in locations) + \
                   quicksum((1 - d[n, l]) * population[n] for n in neighborhoods for l in locations) - \
                   quicksum(d[n, l] for n in neighborhoods for l in locations) + \
                   quicksum(transport_costs_segments[s] * z_nl_s[n, l, s] for n in neighborhoods for l in locations for s in segments)
        
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

        # Constraints: Sustainable energy availability at each facility
        for l in locations:
            model.addCons(
                quicksum(y[n, l] * recycling_rates[n] * f[n, l] for n in neighborhoods) <= energy_availability[l],
                f"Energy_{l}"
            )

        # Constraints: Collection frequency related to demand satisfaction
        for n in neighborhoods:
            for l in locations:
                model.addCons(
                    f[n, l] == population[n] * d[n, l],
                    f"Frequency_Demand_{n}_{l}"
                )
                
        # Piecewise Linear Constraints: Ensure each demand fraction lies within a segment
        for n in neighborhoods:
            for l in locations:
                model.addCons(
                    quicksum(z_nl_s[n, l, s] for s in segments) == 1,
                    f"Segment_{n}_{l}"
                )
                
                for s in segments[:-1]:
                    model.addCons(
                        d[n, l] >= d_breakpoints[s] * z_nl_s[n, l, s],
                        f"LowerBoundSeg_{n}_{l}_{s}"
                    )
                    model.addCons(
                        d[n, l] <= d_breakpoints[s+1] * z_nl_s[n, l, s] + (1 - z_nl_s[n, l, s]) * d_breakpoints[-1],
                        f"UpperBoundSeg_{n}_{l}_{s}"
                    )

        # New Set Covering Constraints
        for n in neighborhoods:
            model.addCons(
                quicksum(y[n, l] for l in locations) >= min_service_centers[n],
                f"Set_Covering_{n}"
            )
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_neighborhoods': 75,
        'potential_locations': 25,
        'population_shape': 25,
        'population_scale': 300,
        'min_population': 2500,
        'max_population': 3000,
        'alpha_recycling_rate': 6.0,
        'beta_recycling_rate': 3.75,
        'number_of_graph_edges': 30,
        'min_distance_multiplier': 1,
        'max_distance_multiplier': 18,
        'facility_cost_shape': 5.0,
        'facility_cost_scale': 2000.0,
        'facility_capacity_shape': 50.0,
        'facility_capacity_scale': 400.0,
        'energy_shape': 0.5,
        'energy_scale': 10000,
        'max_collection_frequency': 50,
        'num_segments': 5,
        'min_transport_cost': 1.0,
        'max_transport_cost': 10.0,
        'min_centers': 2
    }
    
    recycling_optimization = RecyclingOptimization(parameters, seed=seed)
    instance = recycling_optimization.generate_instance()
    solve_status, solve_time = recycling_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")