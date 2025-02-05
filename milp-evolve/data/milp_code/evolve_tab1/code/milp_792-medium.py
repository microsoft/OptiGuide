import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class NetworkOptimizationProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################

    def generate_instance(self):
        # Generating installation costs for routers
        router_install_cost = np.random.randint(self.min_router_install_cost, self.max_router_install_cost, self.number_of_nodes)
        
        # Generating operational connection costs between nodes
        connection_costs = np.random.randint(self.min_connection_cost, self.max_connection_cost, (self.number_of_nodes, self.number_of_nodes))
        
        # Generating router capacities
        router_capacities = np.random.randint(self.min_router_capacity, self.max_router_capacity, self.number_of_nodes)

        # Generating link capacities between routers
        link_capacities = np.random.randint(self.min_link_capacity, self.max_link_capacity, (self.number_of_nodes, self.number_of_nodes))

        # Signal range matrix
        distances = np.random.rand(self.number_of_nodes, self.number_of_nodes)
        signal_ranges = np.where(distances <= self.max_signal_range, 1, 0)

        # Data for new constraints
        maintenance_costs = np.random.randint(self.min_maintenance_cost, self.max_maintenance_cost, self.number_of_nodes)
        resource_requirements = np.random.randint(self.min_resource_requirements, self.max_resource_requirements, self.number_of_nodes)
        
        # Time periods for planning
        time_periods = np.random.randint(self.min_time_periods, self.max_time_periods)
        resource_availability = np.random.randint(self.min_resource_availability, self.max_resource_availability, time_periods)
        
        res = {
            'router_install_cost': router_install_cost,
            'connection_costs': connection_costs,
            'router_capacities': router_capacities,
            'link_capacities': link_capacities,
            'signal_ranges': signal_ranges,
            'maintenance_costs': maintenance_costs,
            'resource_requirements': resource_requirements,
            'time_periods': time_periods,
            'resource_availability': resource_availability,
        }
        return res

    ################# PySCIPOpt Modeling #################

    def solve(self, instance):
        router_install_cost = instance['router_install_cost']
        connection_costs = instance['connection_costs']
        router_capacities = instance['router_capacities']
        link_capacities = instance['link_capacities']
        signal_ranges = instance['signal_ranges']
        maintenance_costs = instance['maintenance_costs']
        resource_requirements = instance['resource_requirements']
        time_periods = instance['time_periods']
        resource_availability = instance['resource_availability']

        number_of_nodes = len(router_install_cost)

        model = Model("NetworkOptimizationProblem")

        # Decision variables
        router_installation = {i: model.addVar(vtype="B", name=f"router_installation_{i}") for i in range(number_of_nodes)}
        link_installation = {(i, j): model.addVar(vtype="B", name=f"link_installation_{i}_{j}") for i in range(number_of_nodes) for j in range(number_of_nodes)}
        
        # Variables for maintenance scheduling and resource planning
        maintenance = {(i, t): model.addVar(vtype="B", name=f"maintenance_{i}_{t}") for i in range(number_of_nodes) for t in range(time_periods)}
        resource_usage = {(i, t): model.addVar(vtype="I", name=f"resource_usage_{i}_{t}") for i in range(number_of_nodes) for t in range(time_periods)}

        # Objective: Minimize total cost (installation costs + connection costs + maintenance costs)
        objective_expr = quicksum(router_install_cost[i] * router_installation[i] for i in range(number_of_nodes))
        objective_expr += quicksum(connection_costs[i][j] * link_installation[(i, j)] for i in range(number_of_nodes) for j in range(number_of_nodes))
        objective_expr += quicksum(maintenance_costs[i] * maintenance[(i, t)] for i in range(number_of_nodes) for t in range(time_periods))

        model.setObjective(objective_expr, "minimize")

        # Constraint: Each node must be covered by at least one router within signal range
        for i in range(number_of_nodes):
            model.addCons(quicksum(router_installation[j] * signal_ranges[i][j] for j in range(number_of_nodes)) >= 1, f"NodeCoverage_{i}")

        # Constraint: Router capacity constraints
        for i in range(number_of_nodes):
            model.addCons(quicksum(link_installation[(i, j)] for j in range(number_of_nodes)) <= router_capacities[i], f"RouterCapacity_{i}")

        # Constraint: Link capacity constraints
        for i in range(number_of_nodes):
            for j in range(number_of_nodes):
                model.addCons(link_installation[(i, j)] <= link_capacities[i][j], f"LinkCapacity_{i}_{j}")

        # New Constraints:

        # Maintenance scheduling
        for i in range(number_of_nodes):
            for t in range(time_periods):
                model.addCons(maintenance[(i, t)] <= router_installation[i], f"MaintenanceScheduling_{i}_{t}")
        
        # Resource constraints for each time period
        for t in range(time_periods):
            model.addCons(quicksum(resource_usage[(i, t)] for i in range(number_of_nodes)) <= resource_availability[t], f"ResourceAvailability_{t}")

        # Resource usage constraints linked to maintenance and installations
        for i in range(number_of_nodes):
            for t in range(time_periods):
                model.addCons(resource_usage[(i, t)] >= maintenance[(i, t)] * resource_requirements[i], f"ResourceUsageLower_{i}_{t}")
                model.addCons(resource_usage[(i, t)] <= maintenance[(i, t)] * resource_requirements[i] + (1 - maintenance[(i, t)]) * resource_requirements[i], f"ResourceUsageUpper_{i}_{t}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_nodes': 225,
        'min_router_install_cost': 750,
        'max_router_install_cost': 3000,
        'min_connection_cost': 180,
        'max_connection_cost': 800,
        'min_router_capacity': 3,
        'max_router_capacity': 300,
        'min_link_capacity': 1350,
        'max_link_capacity': 3000,
        'max_signal_range': 0.38,
        'min_maintenance_cost': 700,
        'max_maintenance_cost': 2500,
        'min_resource_requirements': 5,
        'max_resource_requirements': 700,
        'min_time_periods': 1,
        'max_time_periods': 9,
        'min_resource_availability': 100,
        'max_resource_availability': 3000,
    }
    
    network_optimization = NetworkOptimizationProblem(parameters, seed=seed)
    instance = network_optimization.generate_instance()
    solve_status, solve_time = network_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")