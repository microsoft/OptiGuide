import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class EmergencyShelterResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_customers_emergency, 1) - rand(1, self.n_shelters))**2 +
            (rand(self.n_customers_emergency, 1) - rand(1, self.n_shelters))**2
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers_emergency, self.demand_interval)
        capacities = self.randint(self.n_shelters, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_shelters, self.helipad_construction_cost_interval) * np.sqrt(capacities) +
            self.randint(self.n_shelters, self.medical_equipment_cost_interval)
        )
        transportation_costs = self.unit_transportation_costs() * demands[:, np.newaxis]

        capacities = capacities * self.ratio * np.sum(demands) / np.sum(capacities)
        capacities = np.round(capacities)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs
        }

        # Medical check compliance costs for each shelter
        health_check_costs = self.randint((self.n_shelters, self.n_doctors), self.health_check_cost_interval)

        # Penalty costs for insufficient medical staff
        staff_penalties = self.randint(self.n_shelters, self.staff_penalty_interval)

        res.update({
            'health_check_costs': health_check_costs,
            'staff_penalties': staff_penalties
        })

        n_edges = (self.n_shelters * (self.n_shelters - 1)) // 4
        G = nx.barabasi_albert_graph(self.n_shelters, int(np.ceil(n_edges / self.n_shelters)))
        cliques = list(nx.find_cliques(G))
        random_cliques = [cl for cl in cliques if len(cl) > 2][:self.max_cliques]
        res['cliques'] = random_cliques
        
        # Process times for medical and non-medical checks
        medical_process_times = self.randint(self.n_shelters, self.medical_check_time_interval)
        non_medical_process_times = self.randint(self.n_shelters * self.n_doctors, self.non_medical_check_time_interval)

        res.update({
            'medical_process_times': medical_process_times,
            'non_medical_process_times': non_medical_process_times
        })

        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        health_check_costs = instance['health_check_costs']
        staff_penalties = instance['staff_penalties']
        medical_process_times = instance['medical_process_times']
        non_medical_process_times = instance['non_medical_process_times']
        cliques = instance['cliques']
        
        n_customers_emergency = len(demands)
        n_shelters = len(capacities)
        
        model = Model("EmergencyShelterResourceAllocation")
        
        # Decision variables
        open_shelters = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_shelters)}
        if self.continuous_assignment:
            serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers_emergency) for j in range(n_shelters)}
        else:
            serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers_emergency) for j in range(n_shelters)}
        
        # Additional variables for different medical checks at shelters
        medical_check = {(j, k): model.addVar(vtype="B", name=f"MedicalCheck_{j}_{k}") for j in range(n_shelters) for k in range(self.n_doctors)}

        # Variables for amount of healthy and unhealthy individuals after medical check
        healthy_residents = {j: model.addVar(vtype="C", name=f"HealthyResidents_{j}") for j in range(n_shelters)}
        unhealthy_residents = {j: model.addVar(vtype="C", name=f"UnhealthyResidents_{j}") for j in range(n_shelters)}

        # Objective: minimize the total cost including staff penalties and medical check compliance
        primary_objective_expr = quicksum(fixed_costs[j] * open_shelters[j] for j in range(n_shelters)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers_emergency) for j in range(n_shelters)) + \
                         quicksum(health_check_costs[j, k] * medical_check[j, k] for j in range(n_shelters) for k in range(self.n_doctors)) + \
                         quicksum(staff_penalties[j] * open_shelters[j] for j in range(n_shelters))
                         
        # Secondary objective: maximize the medical check coverage
        secondary_objective_expr = -quicksum(healthy_residents[j] for j in range(n_shelters))

        model.setObjective(primary_objective_expr + (self.secondary_objective_weight * secondary_objective_expr), "minimize")

        # Constraints: each depraved area must be served by at least one shelter
        for i in range(n_customers_emergency):
            model.addCons(quicksum(serve[i, j] for j in range(n_shelters)) >= 1, f"ServingDepravedAreas_{i}")

        # Constraints: capacity limits at each shelter
        for j in range(n_shelters):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers_emergency)) <= capacities[j] * open_shelters[j], f"ShelterCapacity_{j}")
        
        # General constraint on the total number of shelters to be opened
        model.addCons(quicksum(open_shelters[j] for j in range(n_shelters)) <= self.shelter_limit, "NumberOfShelters")
        
        for i in range(n_customers_emergency):
            for j in range(n_shelters):
                model.addCons(serve[i, j] <= open_shelters[j], f"Tightening_{i}_{j}")

        # Constraints: each shelter can have only a specified number of doctors assigned
        for j in range(n_shelters):
            model.addCons(quicksum(medical_check[j, k] for k in range(self.n_doctors)) == open_shelters[j], f"NecessarMedicalCheck_{j}")

        # Constraints for assigning medical staff to shelters within staff limit
        for j in range(n_shelters):
            model.addCons(quicksum(self.number_doctors[k] * medical_check[j, k] for k in range(self.n_doctors)) <= self.medical_staff_limit[j], f"NumberOfMedicalStaff_{j}")

        # Clique constraints
        for idx, clique in enumerate(cliques):
            if len(clique) > 2:
                model.addCons(quicksum(open_shelters[j] for j in clique) <= self.clique_limit, f"Clique_{idx}")

        # New constraint: process time limits for checks
        for j in range(n_shelters):
            model.addCons(healthy_residents[j] * medical_process_times[j] + unhealthy_residents[j] * non_medical_process_times[j] <= self.max_process_time[j], f"ProcessTime_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers_emergency': 100,
        'n_shelters': 100,
        'n_doctors': 3,
        'demand_interval': (5, 36),
        'capacity_interval': (10, 161),
        'helipad_construction_cost_interval': (100, 300),
        'medical_equipment_cost_interval': (50, 100),
        'ratio': 5.0,
        'health_check_cost_interval': (50, 200),
        'staff_penalty_interval': (10, 50),
        'continuous_assignment': True,
        'max_cliques': 10,
        'clique_limit': 3,
        'medical_check_time_interval': (5, 10),
        'non_medical_check_time_interval': (10, 20),
        'secondary_objective_weight': 0.01,
        'max_process_time': [300] * 100,
        'shelter_limit': 20,
        'medical_staff_limit': [15] * 100,
        'number_doctors': [5, 3, 2]
    }

    resource_allocation = EmergencyShelterResourceAllocation(parameters, seed=seed)
    instance = resource_allocation.generate_instance()
    solve_status, solve_time = resource_allocation.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")