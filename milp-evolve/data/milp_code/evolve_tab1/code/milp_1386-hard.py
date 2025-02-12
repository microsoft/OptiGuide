import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class CommunityGraph:
    def __init__(self, number_of_communities, neighbors, capacities):
        self.number_of_communities = number_of_communities
        self.neighbors = neighbors
        self.capacities = capacities

    @staticmethod
    def generate_community(number_of_communities, conn_prob):
        neighbors = {comm: set() for comm in range(number_of_communities)}
        capacities = np.random.randint(50, 200, number_of_communities)
        for comm in range(number_of_communities):
            for other_comm in range(comm + 1, number_of_communities):
                if random.random() < conn_prob:
                    neighbors[comm].add(other_comm)
                    neighbors[other_comm].add(comm)

        return CommunityGraph(number_of_communities, neighbors, capacities)

class MovieScreening:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.Number_of_Communities > 0 and self.Number_of_Movies > 0
        assert self.Movie_Schedule_Cost_Lower_Bound >= 0 and self.Movie_Schedule_Cost_Upper_Bound >= self.Movie_Schedule_Cost_Lower_Bound
        assert self.Min_Community_Capacity > 0 and self.Max_Community_Capacity >= self.Min_Community_Capacity

        community_costs = np.random.randint(self.Min_Community_Cost, self.Max_Community_Cost + 1, self.Number_of_Communities)
        movie_costs = np.random.randint(self.Movie_Schedule_Cost_Lower_Bound, self.Movie_Schedule_Cost_Upper_Bound + 1, (self.Number_of_Communities, self.Number_of_Movies))
        community_capacities = np.random.randint(self.Min_Community_Capacity, self.Max_Community_Capacity + 1, self.Number_of_Communities)
        movie_preferences = np.random.randint(1, 10, self.Number_of_Movies)
        booking_values = np.random.uniform(0.8, 1.0, (self.Number_of_Communities, self.Number_of_Movies))

        scenario_data = [{} for _ in range(self.No_of_Scenarios)]
        for s in range(self.No_of_Scenarios):
            scenario_data[s]['booking'] = {m: max(0, np.random.gamma(movie_preferences[m], movie_preferences[m] * self.Preference_Variation, )) for m in range(self.Number_of_Movies)}

        transport_cost = np.random.uniform(self.Min_Transport_Cost, self.Max_Transport_Cost, (self.Number_of_Communities, self.Number_of_Movies))
        demand = np.random.randint(self.Min_Demand, self.Max_Demand, self.Number_of_Movies)
        factory_capacity = np.random.randint(self.Min_Factory_Capacity, self.Max_Factory_Capacity, self.Number_of_Communities)
        production_cost = np.random.uniform(self.Min_Production_Cost, self.Max_Production_Cost, self.Number_of_Communities)

        return {
            "community_costs": community_costs,
            "movie_costs": movie_costs,
            "community_capacities": community_capacities,
            "movie_preferences": movie_preferences,
            "booking_values": booking_values,
            "scenario_data": scenario_data,
            "transport_cost": transport_cost,
            "demand": demand,
            "factory_capacity": factory_capacity,
            "production_cost": production_cost
        }

    def solve(self, instance):
        community_costs = instance['community_costs']
        movie_costs = instance['movie_costs']
        community_capacities = instance['community_capacities']
        scenario_data = instance['scenario_data']
        booking_values = instance['booking_values']
        transport_cost = instance['transport_cost']
        demand = instance['demand']
        factory_capacity = instance['factory_capacity']
        production_cost = instance['production_cost']

        model = Model("MovieScreening")
        number_of_communities = len(community_costs)
        number_of_movies = len(movie_costs[0])
        no_of_scenarios = len(scenario_data)

        # Decision variables
        community_vars = {c: model.addVar(vtype="B", name=f"Community_{c}") for c in range(number_of_communities)}
        movie_vars = {(c, m): model.addVar(vtype="B", name=f"Community_{c}_Movie_{m}") for c in range(number_of_communities) for m in range(number_of_movies)}
        production_vars = {(c, m): model.addVar(vtype="C", name=f"Production_C_{c}_M_{m}") for c in range(number_of_communities) for m in range(number_of_movies)}

        model.setObjective(
            quicksum(community_costs[c] * community_vars[c] for c in range(number_of_communities)) +
            quicksum(movie_costs[c, m] * movie_vars[c, m] for c in range(number_of_communities) for m in range(number_of_movies)) +
            quicksum(production_cost[c] * community_vars[c] for c in range(number_of_communities)) +
            quicksum(transport_cost[c][m] * production_vars[c, m] for c in range(number_of_communities) for m in range(number_of_movies)) +
            (1 / no_of_scenarios) * quicksum(quicksum(scenario_data[s]['booking'][m] * movie_vars[c, m] for m in range(number_of_movies)) for c in range(number_of_communities) for s in range(no_of_scenarios)), "minimize"
        )

        # Constraints: Each movie must be assigned to exactly one community
        for m in range(number_of_movies):
            model.addCons(quicksum(movie_vars[c, m] for c in range(number_of_communities)) == 1, f"Movie_{m}_Requirement")

        # Constraints: Only selected communities can host movies
        for c in range(number_of_communities):
            for m in range(number_of_movies):
                model.addCons(movie_vars[c, m] <= community_vars[c], f"Community_{c}_Host_{m}")

        # Constraints: Communities cannot exceed their movie capacities in each scenario
        for s in range(no_of_scenarios):
            for c in range(number_of_communities):
                model.addCons(quicksum(scenario_data[s]['booking'][m] * movie_vars[c, m] for m in range(number_of_movies)) <= community_capacities[c] * community_vars[c], f"Community_{c}_Scenario_{s}_Capacity")

        # Constraints: Each community must achieve a minimum booking value
        min_booking = 0.9
        for m in range(number_of_movies):
            model.addCons(quicksum(booking_values[c, m] * movie_vars[c, m] for c in range(number_of_communities)) >= min_booking, f"Movie_{m}_Booking")

        # Constraints: Each community must not exceed the maximum number of movies
        max_movies = 5
        for c in range(number_of_communities):
            model.addCons(quicksum(movie_vars[c, m] for m in range(number_of_movies)) <= community_vars[c] * max_movies, f"Community_{c}_MaxMovies")

        # New Constraints: Production should not exceed capacity
        for c in range(number_of_communities):
            model.addCons(quicksum(production_vars[c, m] for m in range(number_of_movies)) <= factory_capacity[c] * community_vars[c], f"Community_{c}_Capacity")

        # New Constraints: Total production must meet demand
        for m in range(number_of_movies):
            model.addCons(quicksum(production_vars[c, m] for c in range(number_of_communities)) >= demand[m], f"Movie_{m}_Demand")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'Number_of_Communities': 100,
        'Number_of_Movies': 20,
        'Movie_Schedule_Cost_Lower_Bound': 100,
        'Movie_Schedule_Cost_Upper_Bound': 500,
        'Min_Community_Cost': 1000,
        'Max_Community_Cost': 3000,
        'Min_Community_Capacity': 50,
        'Max_Community_Capacity': 150,
        'No_of_Scenarios': 10,
        'Preference_Variation': 0.2,
        'Min_Transport_Cost': 5.0,
        'Max_Transport_Cost': 25.0,
        'Min_Demand': 20,
        'Max_Demand': 120,
        'Min_Factory_Capacity': 500,
        'Max_Factory_Capacity': 1200,
        'Min_Production_Cost': 50.0,
        'Max_Production_Cost': 300.0,
    }

    movie_screening_optimizer = MovieScreening(parameters, seed=42)
    instance = movie_screening_optimizer.generate_instance()
    solve_status, solve_time, objective_value = movie_screening_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")