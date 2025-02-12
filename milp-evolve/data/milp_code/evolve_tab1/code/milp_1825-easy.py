import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class ComplexFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def generate_transportation_costs(self, graph, node_positions, customers, facilities):
        m = len(customers)
        n = len(facilities)
        costs = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                customer_node = customers[i]
                facility_node = facilities[j]
                path_length = nx.shortest_path_length(graph, source=customer_node, target=facility_node, weight='weight')
                costs[i, j] = path_length
        return costs

    def generate_instance(self):
        graph = nx.random_geometric_graph(self.n_nodes, radius=self.radius)
        pos = nx.get_node_attributes(graph, 'pos')
        customers = random.sample(list(graph.nodes), self.n_customers)
        facilities = random.sample(list(graph.nodes), self.n_facilities)
        
        demands = np.random.poisson(self.avg_demand, self.n_customers)
        capacities = self.randint(self.n_facilities, self.capacity_interval)
        fixed_costs = self.randint(self.n_facilities, self.fixed_cost_interval)
        transportation_costs = self.generate_transportation_costs(graph, pos, customers, facilities)

        vehicle_capacities = self.randint(self.n_vehicle_types, self.vehicle_capacity_interval)
        vehicle_costs_per_km = np.random.uniform(self.vehicle_cost_min, self.vehicle_cost_max, self.n_vehicle_types)
        
        res = {
            'demands': demands,
            'capacities': capacities,
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'vehicle_capacities': vehicle_capacities,
            'vehicle_costs_per_km': vehicle_costs_per_km
        }

        factory_costs = np.random.exponential(50, size=self.n_facilities).tolist()
        res['factory_costs'] = factory_costs
        
        demand_scenarios = np.random.normal(demands, scale=self.demand_std_deviation, size=(self.n_scenarios, self.n_customers))
        transportation_cost_scenarios = np.array([self.generate_transportation_costs(graph, pos, customers, facilities) for _ in range(self.n_scenarios)])
        res['demand_scenarios'] = demand_scenarios
        res['transportation_cost_scenarios'] = transportation_cost_scenarios
        
        carbon_emissions_per_km = np.random.uniform(0.1, 0.5, self.n_vehicle_types)
        renewable_energy_costs = np.random.uniform(10, 50, self.n_facilities)
        res['carbon_emissions_per_km'] = carbon_emissions_per_km
        res['renewable_energy_costs'] = renewable_energy_costs
        
        capacity_for_special_treatment = self.randint(self.n_facilities, self.special_treatment_capacity_interval)
        res['capacity_for_special_treatment'] = capacity_for_special_treatment
        peak_period_demand_increase = np.random.uniform(self.peak_period_demand_increase_min, self.peak_period_demand_increase_max, self.n_customers)
        res['peak_period_demand_increase'] = peak_period_demand_increase
        
        multi_period_demands = np.array([demands * (1 + np.random.uniform(-self.multi_period_variation, self.multi_period_variation, self.n_customers)) for _ in range(self.n_periods)])

        res['multi_period_demands'] = multi_period_demands
        
        return res

    def solve(self, instance):
        demands = instance['demands']
        capacities = instance['capacities']
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        vehicle_capacities = instance['vehicle_capacities']
        vehicle_costs_per_km = instance['vehicle_costs_per_km']
        factory_costs = instance['factory_costs']
        demand_scenarios = instance['demand_scenarios']
        transportation_cost_scenarios = instance['transportation_cost_scenarios']
        carbon_emissions_per_km = instance['carbon_emissions_per_km']
        renewable_energy_costs = instance['renewable_energy_costs']
        capacity_for_special_treatment = instance['capacity_for_special_treatment']
        peak_period_demand_increase = instance['peak_period_demand_increase']
        multi_period_demands = instance['multi_period_demands']

        n_customers = len(demands)
        n_facilities = len(capacities)
        n_vehicle_types = len(vehicle_capacities)
        n_scenarios = len(demand_scenarios)
        n_periods = len(multi_period_demands)

        model = Model("ComplexFacilityLocation")

        open_facilities = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(n_facilities)}
        serve = {(i, j): model.addVar(vtype="I", name=f"Serve_{i}_{j}") for i in range(n_customers) for j in range(n_facilities)}
        vehicle_assign = {(j, k): model.addVar(vtype="B", name=f"VehicleAssign_{j}_{k}") for j in range(n_facilities) for k in range(n_vehicle_types)}
        unmet_demand = {i: model.addVar(vtype="I", name=f"UnmetDemand_{i}") for i in range(n_customers)}
        multi_period_serve = {(t, i, j): model.addVar(vtype="C", name=f"MPServe_{t}_{i}_{j}") for t in range(n_periods) for i in range(n_customers) for j in range(n_facilities)}

        fixed_costs_expr = quicksum(fixed_costs[j] * open_facilities[j] for j in range(n_facilities))
        transportation_costs_expr = quicksum(transportation_costs[i, j] * serve[i, j] for i in range(n_customers) for j in range(n_facilities))
        vehicle_costs_expr = quicksum(vehicle_costs_per_km[k] * vehicle_assign[j, k] * capacities[j] for j in range(n_facilities) for k in range(n_vehicle_types))
        factory_costs_expr = quicksum(factory_costs[j] * open_facilities[j] for j in range(n_facilities))
        carbon_emission_costs_expr = quicksum(carbon_emissions_per_km[k] * vehicle_assign[j, k] for j in range(n_facilities) for k in range(n_vehicle_types))
        renewable_energy_costs_expr = quicksum(renewable_energy_costs[j] * open_facilities[j] for j in range(n_facilities))

        unmet_demand_penalty = 1000
        scenario_penalty = 500
        period_variation_penalty = 200

        objective_expr = (
            fixed_costs_expr 
            + transportation_costs_expr 
            + vehicle_costs_expr 
            + factory_costs_expr 
            + carbon_emission_costs_expr
            + renewable_energy_costs_expr
            + quicksum(unmet_demand_penalty * unmet_demand[i] for i in range(n_customers))
            + quicksum(scenario_penalty * (demand_scenarios[s, i] - serve[i, j]) for s in range(n_scenarios) for i in range(n_customers) for j in range(n_facilities))
            + quicksum(period_variation_penalty * (multi_period_demands[t][i] - multi_period_serve[t, i, j]) for t in range(n_periods) for i in range(n_customers) for j in range(n_facilities))
        )

        for i in range(n_customers):
            model.addCons(quicksum(serve[i, j] for j in range(n_facilities)) + unmet_demand[i] == demands[i], f"Demand_{i}")

        for j in range(n_facilities):
            model.addCons(quicksum(serve[i, j] for i in range(n_customers)) <= capacities[j] * open_facilities[j], f"Capacity_{j}")

        for j in range(n_facilities):
            model.addCons(quicksum(vehicle_assign[j, k] * vehicle_capacities[k] for k in range(n_vehicle_types)) >= capacities[j] * open_facilities[j], f"VehicleCapacity_{j}")

        for i in range(n_customers):
            for t in range(n_periods):
                model.addCons(quicksum(multi_period_serve[t, i, j] for j in range(n_facilities)) <= multi_period_demands[t][i], f"MultiPeriodDemand_{t}_{i}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_customers': 55,
        'n_facilities': 74,
        'n_nodes': 84,
        'radius': 0.38,
        'avg_demand': 735,
        'capacity_interval': (112, 1811),
        'fixed_cost_interval': (2500, 2775),
        'vehicle_capacity_interval': (375, 750),
        'vehicle_cost_min': 3.38,
        'vehicle_cost_max': 26.25,
        'n_vehicle_types': 36,
        'demand_std_deviation': 200,
        'n_scenarios': 5,
        'carbon_emission_limit': 5000,
        'renewable_energy_percentage': 0.17,
        'special_treatment_capacity_interval': (112, 900),
        'peak_period_demand_increase_min': 0.8,
        'peak_period_demand_increase_max': 1.3,
        'n_periods': 4,
        'multi_period_variation': 0.15,
    }
    complex_facility_location = ComplexFacilityLocation(parameters, seed=seed)
    instance = complex_facility_location.generate_instance()
    solve_status, solve_time = complex_facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")