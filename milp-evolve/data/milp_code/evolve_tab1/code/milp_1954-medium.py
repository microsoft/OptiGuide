import random
import time
import numpy as np
import networkx as nx
from itertools import permutations
from pyscipopt import Model, quicksum


class City:
    """Helper function: Container for a city graph."""
    def __init__(self, number_of_neighborhoods, streets, traffic_levels):
        self.number_of_neighborhoods = number_of_neighborhoods
        self.neighborhoods = np.arange(number_of_neighborhoods)
        self.streets = streets
        self.traffic_levels = traffic_levels

    @staticmethod
    def generate_city(number_of_neighborhoods, street_prob):
        """Generate a random city graph with traffic levels based on edge probability."""
        edges = set()
        traffic_levels = np.zeros((number_of_neighborhoods, number_of_neighborhoods), dtype=float)
        for edge in permutations(np.arange(number_of_neighborhoods), 2):
            if np.random.uniform() < street_prob:
                edges.add(edge)
                traffic_levels[edge[0], edge[1]] = np.random.uniform(0.5, 1.5)  # Congestion factor
                traffic_levels[edge[1], edge[0]] = traffic_levels[edge[0], edge[1]]  # Symmetric
        city = City(number_of_neighborhoods, edges, traffic_levels)
        return city


class UrbanWasteManagement:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_city(self):
        return City.generate_city(self.n_neighborhoods, self.street_probability)

    def generate_instance(self):
        city = self.generate_city()
        waste_generation = np.random.randint(1, 100, size=city.number_of_neighborhoods)
        travel_costs = np.random.randint(1, 100, size=(city.number_of_neighborhoods, city.number_of_neighborhoods))
        energy_costs = np.random.rand(city.number_of_neighborhoods, city.number_of_neighborhoods) * 10

        max_energy_capacity = np.random.randint(100, 500)
        collection_deadlines = np.random.randint(3, 8, size=city.number_of_neighborhoods)

        num_trucks = 20
        max_truck_capacity = np.random.randint(100, 300, size=num_trucks)
        
        # Additional data for traffic congestion
        avg_speed = np.random.uniform(10, 30)  # km/h
        fuel_cost = np.random.uniform(0.1, 0.5)

        res = {
            'city': city,
            'waste_generation': waste_generation,
            'travel_costs': travel_costs,
            'energy_costs': energy_costs,
            'max_energy_capacity': max_energy_capacity,
            'collection_deadlines': collection_deadlines,
            'num_trucks': num_trucks,
            'max_truck_capacity': max_truck_capacity,
            'avg_speed': avg_speed,
            'fuel_cost': fuel_cost,
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        city = instance['city']
        waste_generation = instance['waste_generation']
        travel_costs = instance['travel_costs']
        energy_costs = instance['energy_costs']
        max_energy_capacity = instance['max_energy_capacity']
        collection_deadlines = instance['collection_deadlines']
        num_trucks = instance['num_trucks']
        max_truck_capacity = instance['max_truck_capacity']
        avg_speed = instance['avg_speed']
        fuel_cost = instance['fuel_cost']

        model = Model("UrbanWasteManagement")

        # Add variables
        truck_vars = {(t, n): model.addVar(vtype="B", name=f"TruckAssignment_{t}_{n}") for t in range(num_trucks) for n in city.neighborhoods}
        energy_vars = {(t, n): model.addVar(vtype="C", name=f"EnergyUsage_{t}_{n}") for t in range(num_trucks) for n in city.neighborhoods}
        early_penalty_vars = {(t, n): model.addVar(vtype="C", name=f"EarlyPenalty_{t}_{n}") for t in range(num_trucks) for n in city.neighborhoods}
        late_penalty_vars = {(t, n): model.addVar(vtype="C", name=f"LatePenalty_{t}_{n}") for t in range(num_trucks) for n in city.neighborhoods}

        # Ensure every neighborhood is served
        for n in city.neighborhoods:
            model.addCons(
                quicksum(truck_vars[t, n] for t in range(num_trucks)) == 1,
                name=f"NeighborhoodServed_{n}"
            )

        # Energy constraints
        for t in range(num_trucks):
            model.addCons(
                quicksum(energy_vars[t, n] for n in city.neighborhoods) <= max_energy_capacity,
                name=f"TruckEnergy_{t}"
            )

        # Traffic and travel constraints
        for t in range(num_trucks):
            for i, j in city.streets:
                travel_time = travel_costs[i, j] / avg_speed * city.traffic_levels[i, j]
                model.addCons(
                    energy_vars[t, j] >= truck_vars[t, i] * travel_time * fuel_cost,
                    name=f"TravelEnergy_{t}_{i}_{j}"
                )

        # Penalties for early/late collection
        for t in range(num_trucks):
            for n in city.neighborhoods:
                deadline = collection_deadlines[n]
                model.addCons(
                    early_penalty_vars[t, n] >= truck_vars[t, n] * max(collection_deadlines) - deadline,
                    name=f"EarlyPenalty_{t}_{n}"
                )
                model.addCons(
                    late_penalty_vars[t, n] >= deadline - truck_vars[t, n] * min(collection_deadlines),
                    name=f"LatePenalty_{t}_{n}"
                )

        # Objective to minimize cost and penalties
        total_cost = quicksum(energy_vars[t, n] * energy_costs[t % len(city.neighborhoods), n] for t in range(num_trucks) for n in city.neighborhoods) + \
                     quicksum(early_penalty_vars[t, n] + late_penalty_vars[t, n] for t in range(num_trucks) for n in city.neighborhoods)

        model.setObjective(total_cost, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_neighborhoods': 50,
        'street_probability': 0.77,
    }

    urban_waste_management = UrbanWasteManagement(parameters, seed=seed)
    instance = urban_waste_management.generate_instance()
    solve_status, solve_time = urban_waste_management.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")