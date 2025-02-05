import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class EmergencyFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_evacuation_travel_time(self, graph, node_positions, victims, facilities):
        m = len(victims)
        n = len(facilities)
        travel_times = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                victim_node = victims[i]
                facility_node = facilities[j]
                path_length = nx.shortest_path_length(graph, source=victim_node, target=facility_node, weight='weight')
                travel_times[i, j] = path_length
        return travel_times

    def find_cliques(self, demand_nodes, demand_threshold):
        """Find all cliques such that the sum of the demands in each clique exceeds the given threshold"""
        node_demand_pairs = [(i, dem) for i, dem in enumerate(demand_nodes)]
        cliques = []
        for i in range(len(node_demand_pairs)):
            for j in range(i + 1, len(node_demand_pairs)):
                if node_demand_pairs[i][1] + node_demand_pairs[j][1] >= demand_threshold:
                    cliques.append([node_demand_pairs[i][0], node_demand_pairs[j][0]])
        return cliques

    def generate_instance(self):
        graph = nx.random_geometric_graph(self.n_nodes, radius=self.radius)
        pos = nx.get_node_attributes(graph, 'pos')
        victims = random.sample(list(graph.nodes), self.n_victims)
        facilities = random.sample(list(graph.nodes), self.n_facilities)
        
        rescue_demand = np.random.poisson(self.avg_rescue_demand, self.n_victims)
        facility_capacities = self.randint(self.n_facilities, self.capacity_interval)
        emergency_opening_cost = self.randint(self.n_facilities, self.opening_cost_interval)
        evacuation_travel_time = self.generate_evacuation_travel_time(graph, pos, victims, facilities)

        vehicle_capacities = self.randint(self.n_vehicle_types, self.vehicle_capacity_interval)
        vehicle_costs_per_km = np.random.uniform(self.vehicle_cost_min, self.vehicle_cost_max, self.n_vehicle_types)
        
        res = {
            'rescue_demand': rescue_demand,
            'facility_capacities': facility_capacities,
            'emergency_opening_cost': emergency_opening_cost,
            'evacuation_travel_time': evacuation_travel_time,
            'vehicle_capacities': vehicle_capacities,
            'vehicle_costs_per_km': vehicle_costs_per_km
        }

        emergency_scenario_costs = np.random.exponential(50, size=self.n_facilities).tolist()
        res['emergency_scenario_costs'] = emergency_scenario_costs
        
        demand_scenarios = np.random.normal(rescue_demand, scale=self.demand_std_deviation, size=(self.n_scenarios, self.n_victims))
        travel_time_scenarios = np.array([self.generate_evacuation_travel_time(graph, pos, victims, facilities) for _ in range(self.n_scenarios)])
        res['demand_scenarios'] = demand_scenarios
        res['travel_time_scenarios'] = travel_time_scenarios
        
        carbon_emissions_per_km = np.random.uniform(0.1, 0.5, self.n_vehicle_types)
        renewable_energy_costs = np.random.uniform(10, 50, self.n_facilities)
        res['carbon_emissions_per_km'] = carbon_emissions_per_km
        res['renewable_energy_costs'] = renewable_energy_costs
        
        capacity_for_special_treatment = self.randint(self.n_facilities, self.special_treatment_capacity_interval)
        res['capacity_for_special_treatment'] = capacity_for_special_treatment
        peak_period_demand_increase = np.random.uniform(self.peak_period_demand_increase_min, self.peak_period_demand_increase_max, self.n_victims)
        res['peak_period_demand_increase'] = peak_period_demand_increase
        
        multi_period_demands = np.array([rescue_demand * (1 + np.random.uniform(-self.multi_period_variation, self.multi_period_variation, self.n_victims)) for _ in range(self.n_periods)])
        res['multi_period_demands'] = multi_period_demands

        res['cliques'] = self.find_cliques(rescue_demand, self.clique_demand_threshold)
        
        return res

    def solve(self, instance):
        rescue_demand = instance['rescue_demand']
        facility_capacities = instance['facility_capacities']
        emergency_opening_cost = instance['emergency_opening_cost']
        evacuation_travel_time = instance['evacuation_travel_time']
        vehicle_capacities = instance['vehicle_capacities']
        vehicle_costs_per_km = instance['vehicle_costs_per_km']
        emergency_scenario_costs = instance['emergency_scenario_costs']
        demand_scenarios = instance['demand_scenarios']
        travel_time_scenarios = instance['travel_time_scenarios']
        carbon_emissions_per_km = instance['carbon_emissions_per_km']
        renewable_energy_costs = instance['renewable_energy_costs']
        capacity_for_special_treatment = instance['capacity_for_special_treatment']
        peak_period_demand_increase = instance['peak_period_demand_increase']
        multi_period_demands = instance['multi_period_demands']
        cliques = instance['cliques']

        n_victims = len(rescue_demand)
        n_facilities = len(facility_capacities)
        n_vehicle_types = len(vehicle_capacities)
        n_scenarios = len(demand_scenarios)
        n_periods = len(multi_period_demands)

        model = Model("EmergencyFacilityLocation")

        open_facilities = {j: model.addVar(vtype="B", name=f"EmergencyFacilityOpen_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="I", name=f"Serve_{i}_{j}") for i in range(n_victims) for j in range(n_facilities)}
        vehicle_assign = {(j, k): model.addVar(vtype="B", name=f"VehicleAssign_{j}_{k}") for j in range(n_facilities) for k in range(n_vehicle_types)}
        unmet_demand = {i: model.addVar(vtype="I", name=f"UnmetDemand_{i}") for i in range(n_victims)}
        multi_period_serve = {(t, i, j): model.addVar(vtype="C", name=f"MPServe_{t}_{i}_{j}") for t in range(n_periods) for i in range(n_victims) for j in range(n_facilities)}

        opening_costs_expr = quicksum(emergency_opening_cost[j] * open_facilities[j] for j in range(n_facilities))
        travel_time_expr = quicksum(evacuation_travel_time[i, j] * serve[i, j] for i in range(n_victims) for j in range(n_facilities))
        vehicle_costs_expr = quicksum(vehicle_costs_per_km[k] * vehicle_assign[j, k] * facility_capacities[j] for j in range(n_facilities) for k in range(n_vehicle_types))
        scenario_costs_expr = quicksum(emergency_scenario_costs[j] * open_facilities[j] for j in range(n_facilities))
        carbon_emission_costs_expr = quicksum(carbon_emissions_per_km[k] * vehicle_assign[j, k] for j in range(n_facilities) for k in range(n_vehicle_types))
        renewable_energy_costs_expr = quicksum(renewable_energy_costs[j] * open_facilities[j] for j in range(n_facilities))

        unmet_demand_penalty = 1000
        scenario_penalty = 500
        period_variation_penalty = 200

        objective_expr = (
            opening_costs_expr 
            + travel_time_expr 
            + vehicle_costs_expr 
            + scenario_costs_expr 
            + carbon_emission_costs_expr
            + renewable_energy_costs_expr
            + quicksum(unmet_demand_penalty * unmet_demand[i] for i in range(n_victims))
            + quicksum(scenario_penalty * (demand_scenarios[s, i] - serve[i, j]) for s in range(n_scenarios) for i in range(n_victims) for j in range(n_facilities))
            + quicksum(period_variation_penalty * (multi_period_demands[t][i] - multi_period_serve[t, i, j]) for t in range(n_periods) for i in range(n_victims) for j in range(n_facilities))
        )

        for i in range(n_victims):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) + unmet_demand[i] == rescue_demand[i], f"Demand_{i}")

        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] for i in range(n_victims)) <= facility_capacities[j] * open_facilities[j], f"Capacity_{j}")

        for j in range(n_facilities):
            model.addCons(quicksum(vehicle_assign[j, k] * vehicle_capacities[k] for k in range(n_vehicle_types)) >= facility_capacities[j] * open_facilities[j], f"VehicleCapacity_{j}")

        for i in range(n_victims):
            for t in range(n_periods):
                model.addCons(quicksum(multi_period_serve[t, i, j] for j in range(n_facilities)) <= multi_period_demands[t][i], f"MultiPeriodDemand_{t}_{i}")

        for clique in cliques:
            for j in range(n_facilities):
                model.addCons(quicksum(serve[i, j] for i in clique) <= facility_capacities[j] * open_facilities[j], f"CliqueCapacity_{'_'.join(map(str,clique))}_{j}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_victims': 41,
        'n_facilities': 55,
        'n_nodes': 840,
        'radius': 0.1,
        'avg_rescue_demand': 2205,
        'capacity_interval': (56, 905),
        'opening_cost_interval': (625, 693),
        'vehicle_capacity_interval': (37, 75),
        'vehicle_cost_min': 1.69,
        'vehicle_cost_max': 1.31,
        'n_vehicle_types': 18,
        'demand_std_deviation': 10,
        'n_scenarios': 0,
        'carbon_emission_limit': 5000,
        'renewable_energy_percentage': 0.31,
        'special_treatment_capacity_interval': (5, 45),
        'peak_period_demand_increase_min': 0.38,
        'peak_period_demand_increase_max': 0.13,
        'n_periods': 0,
        'multi_period_variation': 0.45,
        'clique_demand_threshold': 60,
    }

    emergency_facility_location = EmergencyFacilityLocation(parameters, seed=seed)
    instance = emergency_facility_location.generate_instance()
    solve_status, solve_time = emergency_facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")