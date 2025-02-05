import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class MCPP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_zones(self):
        n_zones = np.random.randint(self.min_z, self.max_z)
        Z = nx.erdos_renyi_graph(n=n_zones, p=self.er_prob, seed=self.seed)
        return Z

    def generate_marketing_costs(self, Z):
        for node in Z.nodes:
            Z.nodes[node]['visibility'] = np.random.randint(1, 100)
        for u, v in Z.edges:
            Z[u][v]['budget'] = (Z.nodes[u]['visibility'] + Z.nodes[v]['visibility']) / float(self.budget_param)

    def generate_optional_events(self, Z):
        Events = set()
        for edge in Z.edges:
            if np.random.random() <= self.alpha:
                Events.add(edge)
        return Events

    def generate_instance(self):
        Z = self.generate_random_zones()
        self.generate_marketing_costs(Z)
        Events = self.generate_optional_events(Z)
        res = {'Z': Z, 'Events': Events}

        max_employee_effort = {node: np.random.lognormal(mean=7, sigma=1.0) for node in Z.nodes}
        volunteer_probabilities = {(u, v): np.random.uniform(0.3, 0.9) for u, v in Z.edges}
        zone_time_windows = {node: (np.random.randint(1, 100), np.random.randint(100, 200)) for node in Z.nodes}
        marketing_costs = {(u, v): np.random.uniform(1.0, 50.0) for u, v in Z.edges}
        earnings_potential = {node: np.random.uniform(50.0, 200.0) for node in Z.nodes}
        
        res.update({
            'max_employee_effort': max_employee_effort,
            'volunteer_probabilities': volunteer_probabilities,
            'zone_time_windows': zone_time_windows,
            'marketing_costs': marketing_costs,
            'earnings_potential': earnings_potential,
        })

        # New Data
        energy_consumption = {node: np.random.uniform(0.5, 2.0) for node in Z.nodes}
        labor_cost = {node: np.random.uniform(10, 50) for node in Z.nodes}
        environmental_impact = {node: np.random.normal(20, 5) for node in Z.nodes}
        time_windows_events = {edge: (np.random.randint(1, 10), np.random.randint(10, 20)) for edge in Z.edges}

        res.update({
            'energy_consumption': energy_consumption,
            'labor_cost': labor_cost,
            'environmental_impact': environmental_impact,
            'time_windows_events': time_windows_events,
        })
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        Z, Events = instance['Z'], instance['Events']
        
        max_employee_effort = instance['max_employee_effort']
        volunteer_probabilities = instance['volunteer_probabilities']
        zone_time_windows = instance['zone_time_windows']
        marketing_costs = instance['marketing_costs']
        earnings_potential = instance['earnings_potential']
        
        # New Data
        energy_consumption = instance['energy_consumption']
        labor_cost = instance['labor_cost']
        environmental_impact = instance['environmental_impact']
        time_windows_events = instance['time_windows_events']
        
        model = Model("MCPP")
        zone_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in Z.nodes}
        event_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in Z.edges}
        
        employee_effort_vars = {node: model.addVar(vtype="C", name=f"EmployeeEffort_{node}") for node in Z.nodes}
        volunteer_scheduling_vars = {(u, v): model.addVar(vtype="B", name=f"VolunteerScheduling_{u}_{v}") for u, v in Z.edges}
        zone_earnings_vars = {node: model.addVar(vtype="I", name=f"ZoneEarnings_{node}") for node in Z.nodes}

        # New Variables
        energy_consumption_vars = {node: model.addVar(vtype="C", name=f"EnergyConsumption_{node}") for node in Z.nodes}
        labor_cost_vars = {node: model.addVar(vtype="C", name=f"LaborCost_{node}") for node in Z.nodes}
        environmental_impact_vars = {node: model.addVar(vtype="C", name=f"EnvironmentalImpact_{node}") for node in Z.nodes}
        event_time_vars = {edge: model.addVar(vtype="C", name=f"EventTime_{edge[0]}_{edge[1]}") for edge in Z.edges}
        
        objective_expr = quicksum(
            Z.nodes[node]['visibility'] * zone_vars[f"x{node}"] + earnings_potential[node] * zone_vars[f"x{node}"]
            for node in Z.nodes
        )

        objective_expr -= quicksum(
            Z[u][v]['budget'] * event_vars[f"y{u}_{v}"]
            for u, v in Events
        )

        # New objective terms
        objective_expr -= quicksum(
            energy_consumption[node] * energy_consumption_vars[node] * self.energy_cost
            for node in Z.nodes
        )

        objective_expr -= quicksum(
            labor_cost[node] * labor_cost_vars[node]
            for node in Z.nodes
        )

        objective_expr -= quicksum(
            environmental_impact[node] * environmental_impact_vars[node]
            for node in Z.nodes
        )

        for u, v in Z.edges:
            if (u, v) in Events:
                model.addCons(
                    zone_vars[f"x{u}"] + zone_vars[f"x{v}"] - event_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    zone_vars[f"x{u}"] + zone_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )

        for node in Z.nodes:
            model.addCons(
                employee_effort_vars[node] <= max_employee_effort[node],
                name=f"MaxEmployeeEffort_{node}"
            )
        
        for u, v in Z.edges:
            model.addCons(
                volunteer_scheduling_vars[(u, v)] <= volunteer_probabilities[(u, v)],
                name=f"VolunteerSchedulingProb_{u}_{v}"
            )
            model.addCons(
                volunteer_scheduling_vars[(u, v)] <= employee_effort_vars[u],
                name=f"VolunteerAssignLimit_{u}_{v}"
            )
        
        for node in Z.nodes:
            model.addCons(
                zone_time_windows[node][0] <= zone_earnings_vars[node],
                name=f"ZoneTimeWindowStart_{node}"
            )
            model.addCons(
                zone_earnings_vars[node] <= zone_time_windows[node][1],
                name=f"ZoneTimeWindowEnd_{node}"
            )

        # New constraints
        for node in Z.nodes:
            model.addCons(
                energy_consumption_vars[node] <= energy_consumption[node],
                name=f"EnergyConsumption_{node}"
            )
        
        for node in Z.nodes:
            model.addCons(
                labor_cost_vars[node] <= labor_cost[node],
                name=f"LaborCost_{node}"
            )
        
        for node in Z.nodes:
            model.addCons(
                environmental_impact_vars[node] <= environmental_impact[node],
                name=f"EnvironmentalImpact_{node}"
            )
        
        for u, v in Z.edges:
            model.addCons(
                time_windows_events[(u, v)][0] <= event_time_vars[(u, v)],
                name=f"EventTimeStart_{u}_{v}"
            )
            model.addCons(
                event_time_vars[(u, v)] <= time_windows_events[(u, v)][1],
                name=f"EventTimeEnd_{u}_{v}"
            )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_z': 37,
        'max_z': 390,
        'er_prob': 0.31,
        'budget_param': 281.25,
        'alpha': 0.8,
        'facility_min_count': 3,
        'facility_max_count': 1250,
        'energy_cost': 0.31,
    }

    mcpp = MCPP(parameters, seed=seed)
    instance = mcpp.generate_instance()
    solve_status, solve_time = mcpp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")