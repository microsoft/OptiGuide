import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainNetworkDesign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        capacities = np.random.randint(self.min_capacity, self.max_capacity, size=self.n_supply_nodes)
        fix_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost, size=self.n_hubs)
        transport_cost_sup_to_hub = np.random.randint(self.min_trans_cost, self.max_trans_cost, 
                                                      size=(self.n_supply_nodes, self.n_hubs))
        transport_cost_hub_to_dem = np.random.randint(self.min_trans_cost, self.max_trans_cost, 
                                                      size=(self.n_hubs, self.n_demand_nodes))
        demands = np.random.randint(self.min_demand, self.max_demand, size=self.n_demand_nodes)

        return {
            "capacities": capacities,
            "fix_costs": fix_costs,
            "transport_cost_sup_to_hub": transport_cost_sup_to_hub,
            "transport_cost_hub_to_dem": transport_cost_hub_to_dem,
            "demands": demands
        }

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        capacities = instance['capacities']
        fix_costs = instance['fix_costs']
        transport_cost_sup_to_hub = instance['transport_cost_sup_to_hub']
        transport_cost_hub_to_dem = instance['transport_cost_hub_to_dem']
        demands = instance['demands']

        model = Model("SupplyChainNetworkDesign")

        # Decision variables
        node_supply_vars = {i: model.addVar(vtype="C", name=f"NodeSupply_{i}") for i in range(len(capacities))}
        route_flow_vars = {(i, j, k): model.addVar(vtype="C", name=f"RouteFlow_{i}_{j}_{k}")
                           for i in range(self.n_supply_nodes) for j in range(self.n_hubs) for k in range(self.n_demand_nodes)}
        hub_selected_vars = {j: model.addVar(vtype="B", name=f"HubSelected_{j}") for j in range(len(fix_costs))}

        penalty_vars = {j: model.addVar(vtype="C", name=f"Penalty_{j}") for j in range(self.n_hubs)}

        # Objective: minimize total cost (activation + transportation + penalties)
        objective_expr = quicksum(fix_costs[j] * hub_selected_vars[j] for j in range(len(fix_costs))) + \
                         quicksum(transport_cost_sup_to_hub[i][j] * route_flow_vars[(i, j, k)]
                                  for i in range(self.n_supply_nodes) for j in range(self.n_hubs) for k in range(self.n_demand_nodes)) + \
                         quicksum(transport_cost_hub_to_dem[j][k] * route_flow_vars[(i, j, k)]
                                  for j in range(self.n_hubs) for k in range(self.n_demand_nodes) for i in range(self.n_supply_nodes)) + \
                         quicksum(penalty_vars[j] for j in range(self.n_hubs))

        # Constraints: Flow conservation at supply nodes
        for i in range(self.n_supply_nodes):
            model.addCons(quicksum(route_flow_vars[(i, j, k)] for j in range(self.n_hubs) for k in range(self.n_demand_nodes)) == node_supply_vars[i], 
                          f"SupplyFlowConservation_{i}")

        # Capacity constraints at supply nodes
        for i in range(self.n_supply_nodes):
            model.addCons(node_supply_vars[i] <= capacities[i], f"CapacityConstraint_{i}")

        # Flow conservation at hubs
        for j in range(self.n_hubs):
            for k in range(self.n_demand_nodes):
                model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes)) == demands[k] * hub_selected_vars[j], 
                              f"HubFlowConservation_{j}_{k}")

        # Ensure demand met at demand nodes
        for k in range(self.n_demand_nodes):
            model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes) for j in range(self.n_hubs)) == demands[k], 
                          f"DemandMet_{k}")

        # New Constraints: Big M formulation on hub capacities
        for j in range(self.n_hubs):
            for k in range(self.n_demand_nodes):
                model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes)) <= self.max_hub_flow * hub_selected_vars[j], 
                              f"MaxHubFlow_{j}_{k}")
                model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes)) <=  self.max_hub_flow + self.overutilization_penalty * penalty_vars[j], 
                              f"OverUtilizationPenalty_{j}_{k}")
                model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes)) >= self.max_hub_flow - self.underutilization_penalty * penalty_vars[j], 
                              f"UnderUtilizationPenalty_{j}_{k}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_supply_nodes': 10,
        'n_hubs': 50,
        'n_demand_nodes': 64,
        'min_capacity': 600,
        'max_capacity': 1600,
        'min_fixed_cost': 350,
        'max_fixed_cost': 600,
        'min_trans_cost': 35,
        'max_trans_cost': 350,
        'min_demand': 30,
        'max_demand': 180,
        'max_hub_flow': 5000,  # New parameter: Max flow through each hub
        'underutilization_penalty': 1000,  # New parameter: Penalty for underutilization of hubs
        'overutilization_penalty': 1000,  # New parameter: Penalty for overutilization of hubs
    }
    
    supply_chain = SupplyChainNetworkDesign(parameters, seed)
    instance = supply_chain.generate_instance()
    solve_status, solve_time = supply_chain.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")