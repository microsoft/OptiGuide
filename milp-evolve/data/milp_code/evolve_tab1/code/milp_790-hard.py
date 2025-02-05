import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DataCenterInfrastructureOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def get_instance(self):
        num_facilities = random.randint(self.min_facilities, self.max_facilities)
        num_clients = random.randint(self.min_clients, self.max_clients)

        # Cost matrices
        construction_cost = np.random.randint(5000, 15000, size=num_facilities)
        operational_cost = np.random.randint(200, 1000, size=num_facilities)
        bandwidth_cost = np.random.randint(5, 20, size=(num_clients, num_facilities))

        # Client requirements
        client_server_demand = np.random.randint(10, 50, size=num_clients)
        
        # Demand uncertainty
        client_demand_uncertainty = np.random.randint(2, 5, size=num_clients)

        # Facilities' capacities
        facility_server_capacity = np.random.randint(200, 1000, size=num_facilities)

        # Bandwidth limits
        bandwidth_limit = np.random.randint(100, 500)

        # Budget constraints
        total_budget = np.random.randint(50000, 100000)

        res = {
            'num_facilities': num_facilities,
            'num_clients': num_clients,
            'construction_cost': construction_cost,
            'operational_cost': operational_cost,
            'bandwidth_cost': bandwidth_cost,
            'client_server_demand': client_server_demand,
            'client_demand_uncertainty': client_demand_uncertainty,
            'facility_server_capacity': facility_server_capacity,
            'bandwidth_limit': bandwidth_limit,
            'total_budget': total_budget
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_facilities = instance['num_facilities']
        num_clients = instance['num_clients']
        construction_cost = instance['construction_cost']
        operational_cost = instance['operational_cost']
        bandwidth_cost = instance['bandwidth_cost']
        client_server_demand = instance['client_server_demand']
        client_demand_uncertainty = instance['client_demand_uncertainty']
        facility_server_capacity = instance['facility_server_capacity']
        bandwidth_limit = instance['bandwidth_limit']
        total_budget = instance['total_budget']

        model = Model("DataCenterInfrastructureOptimization")

        # Variables
        FacilityConstruction = {j: model.addVar(vtype="B", name=f"FacilityConstruction_{j}") for j in range(num_facilities)}
        NetworkBandwidth = {(i, j): model.addVar(vtype="I", name=f"NetworkBandwidth_{i}_{j}") for i in range(num_clients) for j in range(num_facilities)}

        # Objective function: Minimize total costs including construction, operational, and bandwidth costs
        TotalCost = quicksum(FacilityConstruction[j] * (construction_cost[j] + operational_cost[j]) for j in range(num_facilities)) + \
                    quicksum(NetworkBandwidth[i, j] * bandwidth_cost[i, j] for i in range(num_clients) for j in range(num_facilities))

        model.setObjective(TotalCost, "minimize")

        # Robust client demand constraints
        for i in range(num_clients):
            demand_min = client_server_demand[i] - client_demand_uncertainty[i]
            demand_max = client_server_demand[i] + client_demand_uncertainty[i]
            # Ensure bandwidth meets the maximum demand scenario
            model.addCons(quicksum(NetworkBandwidth[i, j] for j in range(num_facilities)) >= demand_min, name=f"client_demand_min_{i}")
            model.addCons(quicksum(NetworkBandwidth[i, j] for j in range(num_facilities)) <= demand_max, name=f"client_demand_max_{i}")

        # Facility capacity constraints
        for j in range(num_facilities):
            model.addCons(quicksum(NetworkBandwidth[i, j] for i in range(num_clients)) <= facility_server_capacity[j], name=f"facility_capacity_{j}")

        # Facility activity constraint
        for j in range(num_facilities):
            model.addCons(FacilityConstruction[j] * sum(client_server_demand) >= quicksum(NetworkBandwidth[i, j] for i in range(num_clients)), name=f"facility_activity_{j}")

        # Budget constraint
        model.addCons(quicksum(FacilityConstruction[j] * sum(client_server_demand) for j in range(num_facilities)) <= total_budget, name="budget_constraint")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facilities': 7,
        'max_facilities': 40,
        'min_clients': 67,
        'max_clients': 3000,
        'demand_uncertainty_factor': 0.38,
    }
    
    optimization = DataCenterInfrastructureOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")