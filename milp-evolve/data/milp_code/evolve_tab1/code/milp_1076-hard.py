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

    def generate_instance(self):
        capacities = np.random.randint(self.min_capacity, self.max_capacity, size=self.n_supply_nodes)
        fix_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost, size=self.n_hubs)
        transport_cost_sup_to_hub = np.random.randint(self.min_trans_cost, self.max_trans_cost, size=(self.n_supply_nodes, self.n_hubs))
        transport_cost_hub_to_dem = np.random.randint(self.min_trans_cost, self.max_trans_cost, size=(self.n_hubs, self.n_demand_nodes))
        demands = np.random.randint(self.min_demand, self.max_demand, size=self.n_demand_nodes)
        queue_time_stop = np.random.randint(self.min_queue_time, self.max_queue_time, size=self.n_bus_stops)
        load_route_bus = np.random.randint(self.min_load, self.max_load, size=(self.n_routes, self.n_buses))
        adjust_route = np.random.choice([0, 1], size=(self.n_routes,))

        renewable_energy = np.random.uniform(self.min_renewable_energy, self.max_renewable_energy, size=self.n_hubs)
        operational_hours = np.random.randint(self.min_operational_hours, self.max_operational_hours, size=self.n_hubs)
        traffic_density = np.random.uniform(self.min_traffic_density, self.max_traffic_density, size=self.n_hubs)
        distance_to_hubs = np.random.uniform(self.min_distance, self.max_distance, size=self.n_hubs)
        commercial_demand_factor = np.random.uniform(self.min_comm_demand_factor, self.max_comm_demand_factor, size=self.n_hubs)

        # New instance data
        cross_docking_costs = np.random.uniform(self.min_cross_docking_costs, self.max_cross_docking_costs, size=self.n_hubs)
        energy_consumption_rates = np.random.uniform(self.min_energy_consumption, self.max_energy_consumption, size=self.n_hubs)
        
        return {
            "capacities": capacities,
            "fix_costs": fix_costs,
            "transport_cost_sup_to_hub": transport_cost_sup_to_hub,
            "transport_cost_hub_to_dem": transport_cost_hub_to_dem,
            "demands": demands,
            "queue_time_stop": queue_time_stop,
            "load_route_bus": load_route_bus,
            "adjust_route": adjust_route,
            "renewable_energy": renewable_energy,
            "operational_hours": operational_hours,
            "traffic_density": traffic_density,
            "distance_to_hubs": distance_to_hubs,
            "commercial_demand_factor": commercial_demand_factor,
            "cross_docking_costs": cross_docking_costs,
            "energy_consumption_rates": energy_consumption_rates
        }

    def solve(self, instance):
        capacities = instance['capacities']
        fix_costs = instance['fix_costs']
        transport_cost_sup_to_hub = instance['transport_cost_sup_to_hub']
        transport_cost_hub_to_dem = instance['transport_cost_hub_to_dem']
        demands = instance['demands']
        queue_time_stop = instance['queue_time_stop']
        load_route_bus = instance['load_route_bus']
        adjust_route = instance['adjust_route']

        renewable_energy = instance['renewable_energy']
        operational_hours = instance['operational_hours']
        traffic_density = instance['traffic_density']
        distance_to_hubs = instance['distance_to_hubs']
        commercial_demand_factor = instance['commercial_demand_factor']
        
        cross_docking_costs = instance['cross_docking_costs']
        energy_consumption_rates = instance['energy_consumption_rates']

        model = Model("SupplyChainNetworkDesign")

        # Decision variables
        node_supply_vars = {i: model.addVar(vtype="C", name=f"NodeSupply_{i}") for i in range(len(capacities))}
        route_flow_vars = {(i, j, k): model.addVar(vtype="C", name=f"RouteFlow_{i}_{j}_{k}")
                           for i in range(self.n_supply_nodes) for j in range(self.n_hubs) for k in range(self.n_demand_nodes)}
        hub_selected_vars = {j: model.addVar(vtype="B", name=f"HubSelected_{j}") for j in range(len(fix_costs))}
        penalty_vars = {j: model.addVar(vtype="C", name=f"Penalty_{j}") for j in range(self.n_hubs)}

        queue_time_vars = {s: model.addVar(vtype="C", name=f"QueueTime_{s}") for s in range(self.n_bus_stops)}
        load_route_vars = {(r, b): model.addVar(vtype="C", name=f"LoadRoute_{r}_{b}") for r in range(self.n_routes) for b in range(self.n_buses)}
        adjust_route_vars = {r: model.addVar(vtype="B", name=f"AdjustRoute_{r}") for r in range(self.n_routes)}

        # New decision variables
        renewable_usage_vars = {j: model.addVar(vtype="C", name=f"RenewableUsage_{j}") for j in range(self.n_hubs)}
        operational_time_vars = {j: model.addVar(vtype="C", name=f"OperationalTime_{j}") for j in range(self.n_hubs)}
        traffic_impact_vars = {j: model.addVar(vtype="C", name=f"TrafficImpact_{j}") for j in range(self.n_hubs)}

        # Convex Hull Variables
        hub_route_vars = {(j, k): model.addVar(vtype="B", name=f"HubRoute_{j}_{k}") for j in range(self.n_hubs) for k in range(self.n_demand_nodes)}
        aux_hub_var = {(j, k): model.addVar(vtype="C", name=f"AuxHub_{j}_{k}") for j in range(self.n_hubs) for k in range(self.n_demand_nodes)}

        objective_expr = quicksum(fix_costs[j] * hub_selected_vars[j] for j in range(len(fix_costs))) + \
                         quicksum(transport_cost_sup_to_hub[i][j] * route_flow_vars[(i, j, k)]
                                  for i in range(self.n_supply_nodes) for j in range(self.n_hubs) for k in range(self.n_demand_nodes)) + \
                         quicksum(transport_cost_hub_to_dem[j][k] * route_flow_vars[(i, j, k)]
                                  for j in range(self.n_hubs) for k in range(self.n_demand_nodes) for i in range(self.n_supply_nodes)) + \
                         quicksum(penalty_vars[j] for j in range(self.n_hubs)) + \
                         quicksum(queue_time_vars[s] for s in range(self.n_bus_stops)) + \
                         quicksum(load_route_vars[(r, b)] for r in range(self.n_routes) for b in range(self.n_buses)) + \
                         quicksum(adjust_route_vars[r] for r in range(self.n_routes)) + \
                         quicksum(renewable_usage_vars[j] for j in range(self.n_hubs)) + \
                         quicksum(operational_time_vars[j] for j in range(self.n_hubs)) + \
                         quicksum(traffic_impact_vars[j] for j in range(self.n_hubs)) + \
                         quicksum(cross_docking_costs[j] * hub_selected_vars[j] for j in range(self.n_hubs)) + \
                         quicksum(energy_consumption_rates[j] * operational_time_vars[j] for j in range(self.n_hubs))

        for i in range(self.n_supply_nodes):
            model.addCons(quicksum(route_flow_vars[(i, j, k)] for j in range(self.n_hubs) for k in range(self.n_demand_nodes)) == node_supply_vars[i], f"SupplyFlowConservation_{i}")

        for i in range(self.n_supply_nodes):
            model.addCons(node_supply_vars[i] <= capacities[i], f"CapacityConstraint_{i}")

        for j in range(self.n_hubs):
            for k in range(self.n_demand_nodes):
                model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes)) == demands[k] * hub_selected_vars[j], f"HubFlowConservation_{j}_{k}")

        for k in range(self.n_demand_nodes):
            model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes) for j in range(self.n_hubs)) == demands[k], f"DemandMet_{k}")

        for j in range(self.n_hubs):
            for k in range(self.n_demand_nodes):
                model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes)) <= self.max_hub_flow * hub_selected_vars[j], f"MaxHubFlow_{j}_{k}")
                model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes)) <= self.max_hub_flow + self.overutilization_penalty * penalty_vars[j], f"OverUtilizationPenalty_{j}_{k}")
                model.addCons(quicksum(route_flow_vars[(i, j, k)] for i in range(self.n_supply_nodes)) >= self.max_hub_flow - self.underutilization_penalty * penalty_vars[j], f"UnderUtilizationPenalty_{j}_{k}")

        for s in range(self.n_bus_stops):
            model.addCons(queue_time_vars[s] <= queue_time_stop[s], f"QueueTimeConstraint_{s}")

        for r in range(self.n_routes):
            for b in range(self.n_buses):
                model.addCons(load_route_vars[(r, b)] <= load_route_bus[r][b], f"LoadBusConstraint_{r}_{b}")

        for r in range(self.n_routes):
            model.addCons(adjust_route_vars[r] == adjust_route[r], f"AdjustRouteConstraint_{r}")

        # New constraints for renewable energy availability
        for j in range(self.n_hubs):
            model.addCons(renewable_usage_vars[j] <= renewable_energy[j] * hub_selected_vars[j], f"RenewableEnergyConstraint_{j}")

        # New constraints for operational times
        for j in range(self.n_hubs):
            model.addCons(operational_time_vars[j] <= operational_hours[j] * hub_selected_vars[j], f"OperationalTimeConstraint_{j}")

        # New constraints for traffic density impact
        for j in range(self.n_hubs):
            model.addCons(traffic_impact_vars[j] <= traffic_density[j] * hub_selected_vars[j], f"TrafficDensityConstraint_{j}")

        # Convex Hull Constraints for hub-route selection
        for j in range(self.n_hubs):
            for k in range(self.n_demand_nodes):
                model.addCons(aux_hub_var[j, k] <= hub_selected_vars[j], f"AuxHubUpper_{j}_{k}")
                model.addCons(aux_hub_var[j, k] <= hub_route_vars[j, k], f"AuxHubLower_{j}_{k}")
                model.addCons(aux_hub_var[j, k] >= hub_selected_vars[j] + hub_route_vars[j, k] - 1, f"AuxHubEqual_{j}_{k}")

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
        'n_bus_stops': 20,
        'n_routes': 10,
        'n_buses': 30,
        'min_capacity': 600,
        'max_capacity': 1600,
        'min_fixed_cost': 350,
        'max_fixed_cost': 600,
        'min_trans_cost': 35,
        'max_trans_cost': 350,
        'min_demand': 30,
        'max_demand': 180,
        'max_hub_flow': 5000,
        'underutilization_penalty': 1000,
        'overutilization_penalty': 1000,
        'min_queue_time': 10,
        'max_queue_time': 50,
        'min_load': 20,
        'max_load': 100,
        'min_renewable_energy': 50,
        'max_renewable_energy': 600,
        'min_operational_hours': 4,
        'max_operational_hours': 24,
        'min_traffic_density': 0.1,
        'max_traffic_density': 1.0,
        'min_distance': 1,
        'max_distance': 5,
        'min_comm_demand_factor': 0.5,
        'max_comm_demand_factor': 2.0,
        'min_cross_docking_costs': 100,
        'max_cross_docking_costs': 500,
        'min_energy_consumption': 0.5,
        'max_energy_consumption': 3.0
    }

    supply_chain = SupplyChainNetworkDesign(parameters, seed)
    instance = supply_chain.generate_instance()
    solve_status, solve_time = supply_chain.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")