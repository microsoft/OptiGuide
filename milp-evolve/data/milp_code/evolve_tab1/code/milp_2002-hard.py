import random
import time
import scipy
import numpy as np
from pyscipopt import Model, quicksum

class EventPlanningAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        nnzrs = int(self.num_areas * self.num_zones * self.density)
        
        indices = np.random.choice(self.num_zones, size=nnzrs)
        indices[:2 * self.num_zones] = np.repeat(np.arange(self.num_zones), 2)
        _, zone_size = np.unique(indices, return_counts=True)

        indices[:self.num_areas] = np.random.permutation(self.num_areas)
        i = 0
        indptr = [0]
        for n in zone_size:
            if i >= self.num_areas:
                indices[i:i + n] = np.random.choice(self.num_areas, size=n, replace=False)
            elif i + n > self.num_areas:
                remaining_areas = np.setdiff1d(np.arange(self.num_areas), indices[i:self.num_areas], assume_unique=True)
                indices[self.num_areas:i + n] = np.random.choice(remaining_areas, size=i + n - self.num_areas, replace=False)
            i += n
            indptr.append(i)

        budget = np.random.randint(self.max_budget, size=self.num_zones) + 1

        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.num_areas, self.num_zones)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        essential_zones = np.random.choice(self.num_zones, self.num_essential_zones, replace=False)
        equipment_costs = np.random.randint(self.equipment_cost_low, self.equipment_cost_high, size=self.num_essential_zones)

        venue_costs = np.random.randint(self.min_venue_cost, self.max_venue_cost, self.num_venues)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost, (self.num_venues, self.num_zones))
        venue_capacities = np.random.randint(self.min_capacity, self.max_capacity, self.num_venues)

        venue_levels = np.random.randint(self.min_venue_level, self.max_venue_level, self.num_venues)

        res = {
            'budget': budget,
            'indptr_csr': indptr_csr,
            'indices_csr': indices_csr,
            'essential_zones': essential_zones,
            'equipment_costs': equipment_costs,
            'venue_costs': venue_costs,
            'transport_costs': transport_costs,
            'venue_capacities': venue_capacities,
            'venue_levels': venue_levels
        }
        return res

    def solve(self, instance):
        budget = instance['budget']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        essential_zones = instance['essential_zones']
        equipment_costs = instance['equipment_costs']
        venue_costs = instance['venue_costs']
        transport_costs = instance['transport_costs']
        venue_capacities = instance['venue_capacities']
        venue_levels = instance['venue_levels']

        model = Model("EventPlanningAllocation")
        zone_vars = {}
        zone_equipment_vars = {}
        venue_vars = {}
        allocation_vars = {}

        for j in range(self.num_zones):
            zone_vars[j] = model.addVar(vtype="B", name=f"zone_{j}", obj=budget[j])

        for idx, j in enumerate(essential_zones):
            zone_equipment_vars[j] = model.addVar(vtype="B", name=f"equipment_{j}", obj=equipment_costs[idx])

        for v in range(self.num_venues):
            venue_vars[v] = model.addVar(vtype="B", name=f"venue_{v}", obj=venue_costs[v])
            for j in range(self.num_zones):
                allocation_vars[(v, j)] = model.addVar(vtype="B", name=f"Venue_{v}_Zone_{j}")

        for area in range(self.num_areas):
            zones = indices_csr[indptr_csr[area]:indptr_csr[area + 1]]
            model.addCons(quicksum(zone_vars[j] for j in zones) >= 1, f"Cover_Area_{area}")

        for j in essential_zones:
            areas_zone_covers = np.where(indices_csr == j)[0]
            for area in areas_zone_covers:
                model.addCons(zone_vars[j] >= zone_equipment_vars[j], f"Essential_Coverage_Area_{area}_Zone_{j}")

        for v in range(self.num_venues):
            model.addCons(quicksum(allocation_vars[(v, j)] for j in range(self.num_zones)) <= venue_capacities[v], f"Venue_{v}_Capacity")
            for j in range(self.num_zones):
                model.addCons(allocation_vars[(v, j)] <= venue_vars[v], f"Venue_{v}_Alloc_{j}")

        objective_expr = quicksum(zone_vars[j] * budget[j] for j in range(self.num_zones)) + \
                         quicksum(zone_equipment_vars[j] * equipment_costs[idx] for idx, j in enumerate(essential_zones)) + \
                         quicksum(venue_costs[v] * venue_vars[v] for v in range(self.num_venues)) + \
                         quicksum(transport_costs[v][j] * allocation_vars[(v, j)] for v in range(self.num_venues) for j in range(self.num_zones))

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_areas': 1125,
        'num_zones': 2250,
        'density': 0.73,
        'max_budget': 7,
        'num_essential_zones': 110,
        'equipment_cost_low': 39,
        'equipment_cost_high': 300,
        'num_venues': 135,
        'min_venue_cost': 135,
        'max_venue_cost': 592,
        'min_transport_cost': 50,
        'max_transport_cost': 1000,
        'min_capacity': 535,
        'max_capacity': 2276,
        'min_venue_level': 10,
        'max_venue_level': 50
    }

    problem = EventPlanningAllocation(parameters, seed=seed)
    instance = problem.generate_instance()
    solve_status, solve_time = problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")