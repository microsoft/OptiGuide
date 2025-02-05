import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class CapacitatedFacilityLocationWithClique:
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
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2 +
            (rand(self.n_customers, 1) - rand(1, self.n_facilities))**2
        )
        return costs

    def generate_instance(self):
        demands = self.randint(self.n_customers, self.demand_interval)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = (
            self.randint(self.n_facilities, self.fixed_cost_scale_interval) * np.sqrt(capacities) +
            self.randint(self.n_facilities, self.fixed_cost_cste_interval)
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

        # Environmental impact costs for each disposal method at each facility
        env_impact_costs = self.randint((self.n_facilities, self.n_disposal_methods), self.env_impact_cost_interval)

        # Penalty costs for disruptions due to natural disasters
        disruption_penalties = self.randint(self.n_facilities, self.disruption_penalty_interval)

        res.update({
            'env_impact_costs': env_impact_costs,
            'disruption_penalties': disruption_penalties
        })

        n_edges = (self.n_facilities * (self.n_facilities - 1)) // 4  # About n_facilities choose 2 divided by 4 
        G = nx.barabasi_albert_graph(self.n_facilities, int(np.ceil(n_edges / self.n_facilities)))
        cliques = list(nx.find_cliques(G))
        random_cliques = [cl for cl in cliques if len(cl) > 2][:self.max_cliques]
        res['cliques'] = random_cliques
        
        # Process times for recycled and regular materials
        recycled_process_times = self.randint(self.n_facilities, self.recycled_process_time_interval)
        regular_process_times = self.randint(self.n_facilities* self.n_disposal_methods, self.regular_process_time_interval)

        res.update({
            'recycled_process_times': recycled_process_times,
            'regular_process_times': regular_process_times
        })
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        env_impact_costs = instance['env_impact_costs']
        disruption_penalties = instance['disruption_penalties']
        recycled_process_times = instance['recycled_process_times']
        regular_process_times = instance['regular_process_times']
        cliques = instance['cliques']
        
        n_customers = len(demands)
        n_facilities = len(capacities)
        
        model = Model("FacilityLocationWithClique")
        
        # Decision variables
        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        if self.continuous_assignment:
            serve = {(i, j): model.addVar(vtype="C", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        else:
            serve = {(i, j): model.addVar(vtype="B", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        
        # Additional variables for different waste disposal methods
        disposal_method = {(j, k): model.addVar(vtype="B", name=f"DisposalMethod_{j}_{k}") for j in range(n_facilities) for k in range(self.n_disposal_methods)}

        # Variables for amount of recycled and regular materials used
        recycled_used = {j: model.addVar(vtype="C", name=f"RecycledUsed_{j}") for j in range(n_facilities)}
        regular_used = {j: model.addVar(vtype="C", name=f"RegularUsed_{j}") for j in range(n_facilities)}
        
        # Objective: minimize the total cost including environmental impact and disruption penalties
        primary_objective_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities)) + \
                         quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities)) + \
                         quicksum(env_impact_costs[j, k] * disposal_method[j, k] for j in range(n_facilities) for k in range(self.n_disposal_methods)) + \
                         quicksum(disruption_penalties[j] * open_facilities[j] for j in range(n_facilities))
                         
        # Secondary objective: maximize the use of recycled materials
        secondary_objective_expr = -quicksum(recycled_used[j] for j in range(n_facilities))
        
        model.setObjective(primary_objective_expr + (self.secondary_objective_weight * secondary_objective_expr), "minimize")

        # Constraints: demand must be met
        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) >= 1, f"Demand_{i}")
        
        # Constraints: capacity limits
        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] * demands[i] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")
        
        # Constraints: tightening constraints
        total_demand = np.sum(demands)
        model.addCons(quicksum(capacities[j] * open_facilities[j] for j in range(n_facilities)) >= total_demand, "TotalDemand")
        
        for i in range(n_customers):
            for j in range(n_facilities):
                model.addCons(serve[i, j] <= open_facilities[j], f"Tightening_{i}_{j}")

        # Constraints: each facility can use only one disposal method
        for j in range(n_facilities):
            model.addCons(quicksum(disposal_method[j, k] for k in range(self.n_disposal_methods)) == open_facilities[j], f"DisposalMethod_{j}")

        # Constraints: regulatory compliance (example constraint)
        for j in range(n_facilities):
            for k in range(self.n_disposal_methods):
                model.addCons(disposal_method[j, k] <= self.regulatory_compliance[k], f"Compliance_{j}_{k}")
                
        # Clique constraints
        for idx, clique in enumerate(cliques):
            if len(clique) > 2:
                model.addCons(quicksum(open_facilities[j] for j in clique) <= self.clique_limit, f"Clique_{idx}")

        # New constraint: process time limits
        for j in range(n_facilities):
            model.addCons(recycled_used[j] * recycled_process_times[j] + regular_used[j] * regular_process_times[j] <= self.max_process_time[j], f"ProcessTime_{j}")

        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 100,
        'n_facilities': 100,
        'n_disposal_methods': 3,
        'demand_interval': (5, 36),
        'capacity_interval': (10, 161),
        'fixed_cost_scale_interval': (100, 111),
        'fixed_cost_cste_interval': (0, 91),
        'ratio': 5.0,
        'env_impact_cost_interval': (50, 200),
        'disruption_penalty_interval': (10, 50),
        'regulatory_compliance': [1, 0, 1],  # Example compliance: only methods 0 and 2 are compliant
        'continuous_assignment': True,
        'max_cliques': 10,  # New parameter for controlling the maximum number of cliques
        'clique_limit': 3,  # New parameter: a limit on how many facilities can be simultaneously open in a clique
        'recycled_process_time_interval': (5, 10), # New parameter for recycled material process times
        'regular_process_time_interval': (10, 20), # New parameter for regular material process times
        'secondary_objective_weight': 0.01, # Weight for the secondary objective
        'max_process_time': [200] * 100  # Example maximum processing times for each facility
    }

    facility_location = CapacitatedFacilityLocationWithClique(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")