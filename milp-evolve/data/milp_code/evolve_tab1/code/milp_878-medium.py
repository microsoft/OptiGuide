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
        
        # Package costs and capacities
        num_packages = 3
        package_cost = np.random.randint(500, 2000, size=num_packages)
        package_capacity = np.random.randint(50, 200, size=num_packages)
        max_packages_per_facility = np.random.randint(1, num_packages + 1, size=num_facilities)
        
        # New Sets for coverage constraints
        num_sets = 10
        set_cost = np.random.randint(1000, 3000, size=num_sets)
        coverage_matrix = np.random.randint(0, 2, size=(num_facilities, num_sets))
        
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
            'total_budget': total_budget,
            'num_packages': num_packages,
            'package_cost': package_cost,
            'package_capacity': package_capacity,
            'max_packages_per_facility': max_packages_per_facility,
            'num_sets': num_sets,
            'set_cost': set_cost,
            'coverage_matrix': coverage_matrix
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
        num_packages = instance['num_packages']
        package_cost = instance['package_cost']
        package_capacity = instance['package_capacity']
        max_packages_per_facility = instance['max_packages_per_facility']
        num_sets = instance['num_sets']
        set_cost = instance['set_cost']
        coverage_matrix = instance['coverage_matrix']

        model = Model("DataCenterInfrastructureOptimization")

        # Variables
        FacilityConstruction = {j: model.addVar(vtype="B", name=f"FacilityConstruction_{j}") for j in range(num_facilities)}
        NetworkBandwidth = {(i, j): model.addVar(vtype="I", name=f"NetworkBandwidth_{i}_{j}") for i in range(num_clients) for j in range(num_facilities)}

        # Piecewise linear operational cost parameter k for each facility
        pwl_segments = {(j, k): (random.uniform(0.8, 1.2), round((operational_cost[j] / num_facilities) * (k+1))) for j in range(num_facilities) for k in range(3)}
        UtilizationSegment = {(j, k): model.addVar(vtype="B", name=f"UtilizationSegment_{j}_{k}") for j in range(num_facilities) for k in range(3)}

        # New variables for packages
        PackageAvailable = {(j, k): model.addVar(vtype="B", name=f"PackageAvailable_{j}_{k}") for j in range(num_facilities) for k in range(num_packages)}
        PackageUsage = {(i, j, k): model.addVar(vtype="I", name=f"PackageUsage_{i}_{j}_{k}") for i in range(num_clients) for j in range(num_facilities) for k in range(num_packages)}
        
        # Variables for set covering constraints
        SetChoice = {s: model.addVar(vtype="B", name=f"SetChoice_{s}") for s in range(num_sets)}

        # Objective function: Minimize total costs including construction, bandwidth, piecewise operational costs, and package costs
        TotalCost = quicksum(FacilityConstruction[j] * construction_cost[j] for j in range(num_facilities)) + \
                    quicksum(NetworkBandwidth[i, j] * bandwidth_cost[i, j] for i in range(num_clients) for j in range(num_facilities)) + \
                    quicksum(UtilizationSegment[j, k] * pwl_segments[j, k][1] for j in range(num_facilities) for k in range(3)) + \
                    quicksum(PackageAvailable[j, k] * package_cost[k] for j in range(num_facilities) for k in range(num_packages)) + \
                    quicksum(SetChoice[s] * set_cost[s] for s in range(num_sets))
        
        model.setObjective(TotalCost, "minimize")

        # Robust client demand constraints
        for i in range(num_clients):
            demand_min = client_server_demand[i] - client_demand_uncertainty[i]
            demand_max = client_server_demand[i] + client_demand_uncertainty[i]
            model.addCons(quicksum(NetworkBandwidth[i, j] for j in range(num_facilities)) >= demand_min, name=f"client_demand_min_{i}")
            model.addCons(quicksum(NetworkBandwidth[i, j] for j in range(num_facilities)) <= demand_max, name=f"client_demand_max_{i}")

        # New facility capacity constraints with piecewise linear segments
        for j in range(num_facilities):
            model.addCons(quicksum(UtilizationSegment[j, k] for k in range(3)) == 1, name=f"utilization_segment_constraint_{j}")
            model.addCons(quicksum(NetworkBandwidth[i, j] for i in range(num_clients)) <= facility_server_capacity[j], name=f"facility_capacity_{j}")

        # Facility activity constraint
        for j in range(num_facilities):
            model.addCons(FacilityConstruction[j] * sum(client_server_demand) >= quicksum(NetworkBandwidth[i, j] for i in range(num_clients)), name=f"facility_activity_{j}")

        # Set covering budget constraint
        for j in range(num_facilities):
            model.addCons(quicksum(coverage_matrix[j, s] * SetChoice[s] for s in range(num_sets)) >= FacilityConstruction[j], name=f"coverage_{j}")

        # New constraints for package selection and usage
        for j in range(num_facilities):
            model.addCons(quicksum(PackageAvailable[j, k] for k in range(num_packages)) <= max_packages_per_facility[j], name=f"package_limit_{j}")

        for i in range(num_clients):
            for j in range(num_facilities):
                model.addCons(quicksum(PackageUsage[i, j, k] * package_capacity[k] for k in range(num_packages)) >= client_server_demand[i], 
                              name=f"package_meet_demand_{i}_{j}")
                model.addCons(quicksum(PackageUsage[i, j, k] for k in range(num_packages)) <= quicksum(PackageAvailable[j, k] for k in range(num_packages)), 
                              name=f"package_usage_constraint_{i}_{j}")

        # Start timing
        start_time = time.time()
        # Optimize model
        model.optimize()
        # End timing
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facilities': 5,
        'max_facilities': 80,
        'min_clients': 250,
        'max_clients': 3000,
        'demand_uncertainty_factor': 0.17,
    }

    optimization = DataCenterInfrastructureOptimization(parameters, seed=seed)
    instance = optimization.get_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")