import random
import time
import numpy as np
import scipy.sparse
from pyscipopt import Model, quicksum

class EventSecurityLogistics:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.total_venues * self.total_security_teams * self.density)

        # compute number of security teams per venue
        indices = np.random.choice(self.total_security_teams, size=nnzrs)  # random security team indexes
        indices[:2 * self.total_security_teams] = np.repeat(np.arange(self.total_security_teams), 2)  # force at least 2 security teams per venue
        _, team_per_venue = np.unique(indices, return_counts=True)

        # for each venue, sample random security teams
        indices[:self.total_venues] = np.random.permutation(self.total_venues)  # force at least 1 security team per venue
        i = 0
        indptr = [0]
        for n in team_per_venue:
            if i >= self.total_venues:
                indices[i:i + n] = np.random.choice(self.total_venues, size=n, replace=False)
            elif i + n > self.total_venues:
                remaining_venues = np.setdiff1d(np.arange(self.total_venues), indices[i:self.total_venues], assume_unique=True)
                indices[self.total_venues:i + n] = np.random.choice(remaining_venues, size=i + n - self.total_venues, replace=False)
            i += n
            indptr.append(i)

        # objective coefficients for essential event cover
        cover_costs = np.random.randint(self.max_cost, size=self.total_security_teams) + 1

        # sparse CSC to sparse CSR matrix
        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.total_venues, self.total_security_teams)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        # Additional data for important events (randomly pick some venues as crucial)
        crucial_events = np.random.choice(self.total_security_teams, self.total_big_events, replace=False)
        essential_cost = np.random.randint(self.essential_cost_low, self.essential_cost_high, size=self.total_big_events)
        
        # Failure probabilities not meaningful here; replaced with event overrun probabilities
        event_overrun_probabilities = np.random.uniform(self.overrun_probability_low, self.overrun_probability_high, self.total_security_teams)
        penalty_costs = np.random.randint(self.penalty_cost_low, self.penalty_cost_high, size=self.total_security_teams)
        
        # New data for logistics and vehicle routing
        vehicle_costs = np.random.randint(self.min_vehicle_cost, self.max_vehicle_cost, self.total_vehicles)
        travel_costs = np.random.randint(self.min_travel_cost, self.max_travel_cost, (self.total_vehicles, self.total_security_teams))
        vehicle_capacities = np.random.randint(self.min_vehicle_capacity, self.max_vehicle_capacity, self.total_vehicles)
        travel_times = np.random.uniform(1, 1.5, (self.total_vehicles, self.total_security_teams))
        essential_breaks = np.random.choice([0, 1], (self.total_vehicles, self.total_time_slots), p=[0.9, 0.1])
        labor_costs = np.random.uniform(0.1, 0.5, self.total_time_slots)

        res = {
            'cover_costs': cover_costs,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'crucial_events': crucial_events,
            'essential_cost': essential_cost,
            'event_overrun_probabilities': event_overrun_probabilities,
            'penalty_costs': penalty_costs,
            'vehicle_costs': vehicle_costs,
            'travel_costs': travel_costs,
            'vehicle_capacities': vehicle_capacities,
            'travel_times': travel_times,
            'essential_breaks': essential_breaks,
            'labor_costs': labor_costs
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        cover_costs = instance['cover_costs']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        crucial_events = instance['crucial_events']
        essential_cost = instance['essential_cost']
        event_overrun_probabilities = instance['event_overrun_probabilities']
        penalty_costs = instance['penalty_costs']
        vehicle_costs = instance['vehicle_costs']
        travel_costs = instance['travel_costs']
        vehicle_capacities = instance['vehicle_capacities']
        travel_times = instance['travel_times']
        essential_breaks = instance['essential_breaks']
        labor_costs = instance['labor_costs']

        model = Model("EventSecurityLogistics")
        var_names = {}
        activate_essential = {}
        overrun_vars = {}
        vehicle_vars = {}
        routing_vars = {}
        break_vars = {}

        # Create variables and set objective for classic event cover
        for j in range(self.total_security_teams):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=cover_costs[j])
            overrun_vars[j] = model.addVar(vtype="B", name=f"r_{j}")

        # Additional variables for essential event activation
        for idx, j in enumerate(crucial_events):
            activate_essential[j] = model.addVar(vtype="B", name=f"y_{j}", obj=essential_cost[idx])

        # Vehicle routing variables
        for v in range(self.total_vehicles):
            vehicle_vars[v] = model.addVar(vtype="B", name=f"Vehicle_{v}", obj=vehicle_costs[v])
            for j in range(self.total_security_teams):
                routing_vars[(v, j)] = model.addVar(vtype="B", name=f"Vehicle_{v}_Team_{j}")

        # Break variables
        for v in range(self.total_vehicles):
            for t in range(self.total_time_slots):
                break_vars[(v, t)] = model.addVar(vtype="B", name=f"Vehicle_{v}_TimeSlot_{t}")

        # Add constraints to ensure each venue is covered
        for venue in range(self.total_venues):
            teams = indices_csr[indptr_csr[venue]:indptr_csr[venue + 1]]
            model.addCons(quicksum(var_names[j] - overrun_vars[j] for j in teams) >= 1, f"coverage_{venue}")

        # Ensure prioritized events (essential events) have higher coverage conditions
        for j in crucial_events:
            venues_impacting_j = np.where(indices_csr == j)[0]
            for venue in venues_impacting_j:
                model.addCons(var_names[j] >= activate_essential[j], f"essential_coverage_venue_{venue}_team_{j}")

        # Vehicle capacity and routing constraints
        for v in range(self.total_vehicles):
            model.addCons(quicksum(routing_vars[(v, j)] for j in range(self.total_security_teams)) <= vehicle_capacities[v], f"Vehicle_{v}_Capacity")
            for j in range(self.total_security_teams):
                model.addCons(routing_vars[(v, j)] <= vehicle_vars[v], f"Vehicle_{v}_Route_{j}")
        
        # Break and labor constraints
        for v in range(self.total_vehicles):
            for t in range(self.total_time_slots):
                model.addCons(break_vars[(v, t)] <= vehicle_vars[v], f"Break_Vehicle_{v}_TimeSlot_{t}")
                model.addCons(break_vars[(v, t)] <= (1 - essential_breaks[v, t]), f"Break_Scheduled_Vehicle_{v}_TimeSlot_{t}")

        # Objective: Minimize total cost including penalties for overruns, fixed costs, travel costs, and labor costs
        objective_expr = quicksum(var_names[j] * cover_costs[j] for j in range(self.total_security_teams)) + \
                         quicksum(activate_essential[j] * essential_cost[idx] for idx, j in enumerate(crucial_events)) + \
                         quicksum(overrun_vars[j] * penalty_costs[j] for j in range(self.total_security_teams)) + \
                         quicksum(vehicle_costs[v] * vehicle_vars[v] for v in range(self.total_vehicles)) + \
                         quicksum(travel_costs[v][j] * routing_vars[(v, j)] * travel_times[v][j] for v in range(self.total_vehicles) for j in range(self.total_security_teams)) + \
                         quicksum(labor_costs[t] * break_vars[(v, t)] for v in range(self.total_vehicles) for t in range(self.total_time_slots))
        
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'total_venues': 562,
        'total_security_teams': 1500,
        'density': 0.52,
        'max_cost': 15,
        'total_big_events': 45,
        'essential_cost_low': 270,
        'essential_cost_high': 2000,
        'overrun_probability_low': 0.38,
        'overrun_probability_high': 0.73,
        'penalty_cost_low': 375,
        'penalty_cost_high': 562,
        'total_vehicles': 6,
        'min_vehicle_cost': 90,
        'max_vehicle_cost': 790,
        'min_travel_cost': 81,
        'max_travel_cost': 1715,
        'min_vehicle_capacity': 714,
        'max_vehicle_capacity': 759,
        'total_time_slots': 2,
    }

    event_security_planning = EventSecurityLogistics(parameters, seed=seed)
    instance = event_security_planning.generate_instance()
    solve_status, solve_time = event_security_planning.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")