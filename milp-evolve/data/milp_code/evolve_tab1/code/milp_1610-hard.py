import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyShelterAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_shelters > 0 and self.n_locations > 0
        assert self.min_op_cost >= 0 and self.max_op_cost >= self.min_op_cost
        assert self.min_volunteer_cost >= 0 and self.max_volunteer_cost >= self.min_volunteer_cost
        assert self.min_shelter_capacity > 0 and self.max_shelter_capacity >= self.min_shelter_capacity

        operation_costs = np.random.randint(self.min_op_cost, self.max_op_cost + 1, self.n_shelters)
        volunteer_costs = np.random.randint(self.min_volunteer_cost, self.max_volunteer_cost + 1, (self.n_shelters, self.n_locations))
        shelter_capacities = np.random.randint(self.min_shelter_capacity, self.max_shelter_capacity + 1, self.n_shelters)
        volunteer_requirements = np.random.randint(1, 20, self.n_locations)
        budget_limits = np.random.uniform(self.min_budget_limit, self.max_budget_limit, self.n_shelters)
        distances = np.random.uniform(0, self.max_dist, (self.n_shelters, self.n_locations))

        G = nx.DiGraph()
        node_pairs = []
        for s in range(self.n_shelters):
            for l in range(self.n_locations):
                G.add_edge(f"shelter_{s}", f"location_{l}")
                node_pairs.append((f"shelter_{s}", f"location_{l}"))
                
        return {
            "operation_costs": operation_costs,
            "volunteer_costs": volunteer_costs,
            "shelter_capacities": shelter_capacities,
            "volunteer_requirements": volunteer_requirements,
            "budget_limits": budget_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        operation_costs = instance['operation_costs']
        volunteer_costs = instance['volunteer_costs']
        shelter_capacities = instance['shelter_capacities']
        volunteer_requirements = instance['volunteer_requirements']
        budget_limits = instance['budget_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("EmergencyShelterAllocation")
        n_shelters = len(operation_costs)
        n_locations = len(volunteer_costs[0])
        
        # Decision variables
        EmergencyShelter_vars = {e: model.addVar(vtype="B", name=f"EmergencyShelter_{e}") for e in range(n_shelters)}
        Volunteer_vars = {(u, v): model.addVar(vtype="C", name=f"Volunteer_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total cost including shelter operation costs and volunteer transportation costs.
        model.setObjective(
            quicksum(operation_costs[e] * EmergencyShelter_vars[e] for e in range(n_shelters)) +
            quicksum(volunteer_costs[int(u.split('_')[1]), int(v.split('_')[1])] * Volunteer_vars[(u, v)] for (u, v) in node_pairs),
            "minimize"
        )

        # Volunteer distribution constraint for each location
        for l in range(n_locations):
            model.addCons(
                quicksum(Volunteer_vars[(u, f"location_{l}")] for u in G.predecessors(f"location_{l}")) == volunteer_requirements[l], 
                f"Location_{l}_VolunteerDistribution"
            )

        # Constraints: Locations only receive volunteers if the shelters are open and within budget
        for e in range(n_shelters):
            for l in range(n_locations):
                model.addCons(
                    Volunteer_vars[(f"shelter_{e}", f"location_{l}")] <= budget_limits[e] * EmergencyShelter_vars[e], 
                    f"Shelter_{e}_BudgetLimit_{l}"
                )

        # Constraints: Shelters cannot exceed their capacities
        for e in range(n_shelters):
            model.addCons(
                quicksum(Volunteer_vars[(f"shelter_{e}", f"location_{l}")] for l in range(n_locations)) <= shelter_capacities[e], 
                f"Shelter_{e}_CapacityLimit"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_shelters': 200,
        'n_locations': 120,
        'min_volunteer_cost': 250,
        'max_volunteer_cost': 750,
        'min_op_cost': 1000,
        'max_op_cost': 3000,
        'min_shelter_capacity': 2500,
        'max_shelter_capacity': 2500,
        'min_budget_limit': 5000,
        'max_budget_limit': 10000,
        'max_dist': 3000,
    }

    optimizer = EmergencyShelterAllocation(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")