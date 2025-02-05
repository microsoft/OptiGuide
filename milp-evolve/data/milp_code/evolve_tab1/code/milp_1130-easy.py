import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class ComplexFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_neighborhoods >= self.n_facilities
        assert self.min_fixed_cost >= 0 and self.max_fixed_cost >= self.min_fixed_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_capacity > 0 and self.max_capacity >= self.min_capacity
        
        fixed_costs = np.random.randint(self.min_fixed_cost, self.max_fixed_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_neighborhoods))
        capacities = np.random.randint(self.min_capacity, self.max_capacity + 1, self.n_facilities)
        financial_rewards = np.random.uniform(10, 100, self.n_neighborhoods)
        demand_fluctuation = np.random.normal(1, 0.4, self.n_neighborhoods).tolist()
        ordered_neighborhoods = list(np.random.permutation(self.n_neighborhoods))

        dynamic_traffic_congestion = np.random.uniform(0.5, 2.5, (self.n_facilities, self.n_neighborhoods))
        electricity_prices = np.random.uniform(0.1, 0.5, self.n_time_slots)
        complex_maintenance_schedules = np.random.choice([0, 1], (self.n_facilities, self.n_time_slots), p=[0.8, 0.2])
        carbon_emission_factors = np.random.uniform(0.5, 2.0, (self.n_facilities, self.n_neighborhoods))
        labor_costs = np.random.uniform(25, 75, self.n_time_slots)
        
        environmental_factors = np.random.uniform(0.3, 0.7, self.n_neighborhoods) * self.environmental_factor
        logistical_factors = np.random.uniform(1, 1.2, (self.n_facilities, self.n_neighborhoods))

        # New data generation for combined problem
        emergency_response_times = np.random.randint(5, 30, self.n_neighborhoods)
        node_transport_costs = np.random.randint(20, 100, (self.n_neighborhoods, self.n_neighborhoods))
        traffic_conditions = np.random.randint(1, 10, (self.n_neighborhoods, self.n_neighborhoods))

        return {
            "fixed_costs": fixed_costs,
            "transport_costs": transport_costs,
            "capacities": capacities,
            "financial_rewards": financial_rewards,
            "demand_fluctuation": demand_fluctuation,
            "ordered_neighborhoods": ordered_neighborhoods,
            "dynamic_traffic_congestion": dynamic_traffic_congestion,
            "electricity_prices": electricity_prices,
            "complex_maintenance_schedules": complex_maintenance_schedules,
            "carbon_emission_factors": carbon_emission_factors,
            "labor_costs": labor_costs,
            "environmental_factors": environmental_factors,
            "logistical_factors": logistical_factors,
            "emergency_response_times": emergency_response_times,
            "node_transport_costs": node_transport_costs,
            "traffic_conditions": traffic_conditions
        }

    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        financial_rewards = instance['financial_rewards']
        demand_fluctuation = instance['demand_fluctuation']
        ordered_neighborhoods = instance['ordered_neighborhoods']
        dynamic_traffic_congestion = instance['dynamic_traffic_congestion']
        electricity_prices = instance['electricity_prices']
        complex_maintenance_schedules = instance['complex_maintenance_schedules']
        carbon_emission_factors = instance['carbon_emission_factors']
        labor_costs = instance['labor_costs']
        environmental_factors = instance['environmental_factors']
        logistical_factors = instance['logistical_factors']
        emergency_response_times = instance['emergency_response_times']
        node_transport_costs = instance['node_transport_costs']
        traffic_conditions = instance['traffic_conditions']

        model = Model("ComplexFacilityLocation")
        n_facilities = len(fixed_costs)
        n_neighborhoods = len(transport_costs[0])

        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        allocation_vars = {(f, n): model.addVar(vtype="C", lb=0, ub=1, name=f"Facility_{f}_Neighborhood_{n}") for f in range(n_facilities) for n in range(n_neighborhoods)}
        time_slot_vars = {(f, t): model.addVar(vtype="B", name=f"Facility_{f}_TimeSlot_{t}") for f in range(n_facilities) for t in range(self.n_time_slots)}
        
        # New variables
        emergency_team_vars = {n: model.addVar(vtype="B", name=f"EmergencyTeam_{n}") for n in range(n_neighborhoods)}
        transport_between_neighbors = {(i, j): model.addVar(vtype="B", name=f"Transport_{i}_to_{j}") for i in range(n_neighborhoods) for j in range(n_neighborhoods)}
        transport_time_vars = {(i, j): model.addVar(vtype="C", name=f"TransportTime_{i}_to_{j}") for i in range(n_neighborhoods) for j in range(n_neighborhoods)}

        model.setObjective(
            quicksum(financial_rewards[n] * allocation_vars[f, n] * demand_fluctuation[n] * environmental_factors[n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(fixed_costs[f] * facility_vars[f] for f in range(n_facilities)) -
            quicksum(transport_costs[f][n] * allocation_vars[f, n] * dynamic_traffic_congestion[f][n] * logistical_factors[f][n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(electricity_prices[t] * time_slot_vars[f, t] for f in range(n_facilities) for t in range(self.n_time_slots)) -
            quicksum(carbon_emission_factors[f][n] * allocation_vars[f, n] for f in range(n_facilities) for n in range(n_neighborhoods)) -
            quicksum(labor_costs[t] * time_slot_vars[f, t] for f in range(n_facilities) for t in range(self.n_time_slots)) -
            quicksum(emergency_response_times[n] * emergency_team_vars[n] for n in range(n_neighborhoods)) -
            quicksum(node_transport_costs[i][j] * transport_between_neighbors[i, j] for i in range(n_neighborhoods) for j in range(n_neighborhoods)),
            "maximize"
        )

        for n in range(n_neighborhoods):
            model.addCons(quicksum(allocation_vars[f, n] for f in range(n_facilities)) == 1, f"Neighborhood_{n}_Assignment")
        
        for f in range(n_facilities):
            model.addCons(quicksum(allocation_vars[f, n] for n in range(n_neighborhoods)) <= facility_vars[f] * capacities[f], f"Facility_{f}_Capacity")
            for t in range(self.n_time_slots):
                model.addCons(time_slot_vars[f, t] <= facility_vars[f], f"Maintenance_Facility_{f}_TimeSlot_{t}")
                model.addCons(time_slot_vars[f, t] <= (1 - complex_maintenance_schedules[f][t]), f"Maintenance_Scheduled_Facility_{f}_TimeSlot_{t}")

        for n in range(n_neighborhoods):
            for f in range(n_facilities):
                model.addCons(allocation_vars[f, n] * demand_fluctuation[n] <= capacities[f], f"DemandCapacity_{f}_{n}")

        for i in range(n_neighborhoods - 1):
            n1 = ordered_neighborhoods[i]
            n2 = ordered_neighborhoods[i + 1]
            for f in range(n_facilities):
                model.addCons(allocation_vars[f, n1] + allocation_vars[f, n2] <= 1, f"SOS_Constraint_Facility_{f}_Neighborhoods_{n1}_{n2}")

        for f in range(n_facilities):
            for n in range(n_neighborhoods):
                if transport_costs[f, n] > 0:
                    model.addCons(allocation_vars[f, n] <= facility_vars[f], f"Route_Facility_{f}_Neighborhood_{n}")
        
        # New constraints for emergency team allocation
        model.addCons(quicksum(emergency_team_vars[n] for n in range(n_neighborhoods)) >= self.min_emergency_teams, "MinEmergencyTeams")
        
        for i in range(n_neighborhoods):
            for j in range(n_neighborhoods):
                model.addCons(transport_time_vars[i, j] == transport_between_neighbors[i, j] * traffic_conditions[i, j], f"TransportTime_Constraint_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 87,
        'n_neighborhoods': 189,
        'min_fixed_cost': 374,
        'max_fixed_cost': 1185,
        'min_transport_cost': 240,
        'max_transport_cost': 2058,
        'min_capacity': 238,
        'max_capacity': 2530,
        'n_time_slots': 300,
        'environmental_factor': 0.38,
        'labor_cost_factor': 0.69,
        'budget_factor': 0.8,
        'min_emergency_teams': 25,
    }

    location_optimizer = ComplexFacilityLocation(parameters, seed=42)
    instance = location_optimizer.generate_instance()
    solve_status, solve_time, objective_value = location_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")