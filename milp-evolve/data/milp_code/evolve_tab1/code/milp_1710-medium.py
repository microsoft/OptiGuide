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
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        Z, Events = instance['Z'], instance['Events']
        
        max_employee_effort = instance['max_employee_effort']
        volunteer_probabilities = instance['volunteer_probabilities']
        zone_time_windows = instance['zone_time_windows']
        marketing_costs = instance['marketing_costs']
        earnings_potential = instance['earnings_potential']
        
        model = Model("MCPP")
        zone_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in Z.nodes}
        event_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in Z.edges}
        
        employee_effort_vars = {node: model.addVar(vtype="C", name=f"EmployeeEffort_{node}") for node in Z.nodes}
        volunteer_scheduling_vars = {(u, v): model.addVar(vtype="B", name=f"VolunteerScheduling_{u}_{v}") for u, v in Z.edges}
        zone_earnings_vars = {node: model.addVar(vtype="I", name=f"ZoneEarnings_{node}") for node in Z.nodes}
        
        objective_expr = quicksum(
            Z.nodes[node]['visibility'] * zone_vars[f"x{node}"] + earnings_potential[node] * zone_vars[f"x{node}"]
            for node in Z.nodes
        )

        objective_expr -= quicksum(
            Z[u][v]['budget'] * event_vars[f"y{u}_{v}"]
            for u, v in Events
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

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_z': 75,
        'max_z': 130,
        'er_prob': 0.45,
        'budget_param': 375.0,
        'alpha': 0.59,
        'facility_min_count': 7,
        'facility_max_count': 125,
    }

    mcpp = MCPP(parameters, seed=seed)
    instance = mcpp.generate_instance()
    solve_status, solve_time = mcpp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")