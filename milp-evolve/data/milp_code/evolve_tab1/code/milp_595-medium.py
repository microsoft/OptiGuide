import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class DataCenterPlacementOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        num_data_centers = random.randint(self.min_data_centers, self.max_data_centers)
        num_nodes = random.randint(self.min_nodes, self.max_nodes)

        # Cost matrices
        node_connection_costs = np.random.randint(50, 300, size=(num_nodes, num_data_centers))
        operational_costs = np.random.randint(1000, 5000, size=num_data_centers)

        # Node demands
        nodal_demand = np.random.randint(100, 500, size=num_nodes)

        # MegaServer capacity
        mega_server_capacity = np.random.randint(1000, 5000, size=num_data_centers)

        # New Data: ReliabilityCosts, Peak Demand and Environmental Penalties
        reliability_costs = np.random.rand(num_nodes, num_data_centers) * 50  # Random reliability costs
        
        peak_demand_periods = random.randint(2, 5)
        high_demand_rates = np.random.uniform(1.5, 3.0, size=peak_demand_periods).tolist()
        maintenance_periods = np.random.randint(0, 5, size=num_data_centers)

        environmental_penalties = np.random.uniform(0, 50, size=num_data_centers).tolist()
        renewable_energy = np.random.uniform(0, 1, size=num_data_centers).tolist()

        res = {
            'num_data_centers': num_data_centers,
            'num_nodes': num_nodes,
            'node_connection_costs': node_connection_costs,
            'operational_costs': operational_costs,
            'nodal_demand': nodal_demand,
            'mega_server_capacity': mega_server_capacity,
            'reliability_costs': reliability_costs,
            'high_demand_rates': high_demand_rates,
            'maintenance_periods': maintenance_periods,
            'environmental_penalties': environmental_penalties,
            'renewable_energy': renewable_energy,
            'peak_demand_periods': peak_demand_periods
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        num_data_centers = instance['num_data_centers']
        num_nodes = instance['num_nodes']
        node_connection_costs = instance['node_connection_costs']
        operational_costs = instance['operational_costs']
        nodal_demand = instance['nodal_demand']
        mega_server_capacity = instance['mega_server_capacity']
        reliability_costs = instance['reliability_costs']
        high_demand_rates = instance['high_demand_rates']
        maintenance_periods = instance['maintenance_periods']
        environmental_penalties = instance['environmental_penalties']
        renewable_energy = instance['renewable_energy']
        peak_demand_periods = instance['peak_demand_periods']

        demand_periods = range(peak_demand_periods)

        model = Model("DataCenterPlacementOptimization")

        # Variables
        mega_server = {i: model.addVar(vtype="B", name=f"mega_server_{i}") for i in range(num_data_centers)}
        node_connection = {(i, j): model.addVar(vtype="B", name=f"node_connection_{i}_{j}") for i in range(num_nodes) for j in range(num_data_centers)}
        connection_reliability = {i: model.addVar(vtype="C", name=f"connection_reliability_{i}") for i in range(num_nodes)}
        high_rate_vars = {(i, p): model.addVar(vtype="C", name=f"HighRate_{i}_{p}", lb=0) for i in range(num_nodes) for p in demand_periods}
        medium_rate_vars = {(i, p): model.addVar(vtype="C", name=f"MediumRate_{i}_{p}", lb=0) for i in range(num_nodes) for p in demand_periods}
        low_rate_vars = {(i, p): model.addVar(vtype="C", name=f"LowRate_{i}_{p}", lb=0) for i in range(num_nodes) for p in demand_periods}

        # Objective function: Minimize total cost and maximize reliability
        total_cost = quicksum(node_connection[i, j] * (node_connection_costs[i, j] + reliability_costs[i, j]) for i in range(num_nodes) for j in range(num_data_centers)) + \
                     quicksum(mega_server[j] * operational_costs[j] for j in range(num_data_centers)) + \
                     quicksum(environmental_penalties[j] * (1 - renewable_energy[j]) for j in range(num_data_centers))

        model.setObjective(total_cost, "minimize")

        # Constraints
        for i in range(num_nodes):
            model.addCons(quicksum(node_connection[i, j] for j in range(num_data_centers)) == 1, name=f"node_connection_{i}")

        # Logical constraints: A data center can only connect nodes if it has a mega server
        for j in range(num_data_centers):
            for i in range(num_nodes):
                model.addCons(node_connection[i, j] <= mega_server[j], name=f"data_center_node_{i}_{j}")

        # Connection reliability and capacity constraints
        for j in range(num_data_centers):
            model.addCons(quicksum(node_connection[i, j] * nodal_demand[i] for i in range(num_nodes)) <= mega_server_capacity[j], name=f"mega_server_capacity_{j}")
            for i in range(num_nodes):
                model.addCons(connection_reliability[i] >= 1 - reliability_costs[i, j] * node_connection[i, j], name=f"connection_reliability_{i}")

        # Ensure maintenance periods are considered
        for j in range(num_data_centers):
            if maintenance_periods[j] > 0:
                model.addCons(mega_server[j] == 0, name=f"maintenance_{j}")

        # New constraints for piecewise linear handling of demands
        for i in range(num_nodes):
            for p in demand_periods:
                if p == 0:
                    model.addCons(high_rate_vars[i, p] 
                                  + medium_rate_vars[i, p] 
                                  + low_rate_vars[i, p] == node_connection[i, p], f"RatePiecewise_{i}_{p}")
                else:
                    model.addCons(high_rate_vars[i, p] 
                                  + medium_rate_vars[i, p] 
                                  + low_rate_vars[i, p] == node_connection[i, p] - node_connection[i, p-1], f"RatePiecewise_{i}_{p}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_data_centers': 5,
        'max_data_centers': 900,
        'min_nodes': 5,
        'max_nodes': 300,
        'peak_demand_periods': 30,
    }

    optimization = DataCenterPlacementOptimization(parameters, seed=seed)
    instance = optimization.generate_instance()
    solve_status, solve_time = optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")