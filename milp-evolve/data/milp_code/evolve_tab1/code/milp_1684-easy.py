import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EPLO:
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

    def generate_resources_costs(self, Z):
        for node in Z.nodes:
            Z.nodes[node]['importance'] = np.random.randint(1, 100)
        for u, v in Z.edges:
            Z[u][v]['equipment_cost'] = (Z.nodes[u]['importance'] + Z.nodes[v]['importance']) / float(self.equipment_param)

    def generate_optional_equipment(self, Z):
        E2 = set()
        for edge in Z.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def generate_instance(self):
        Z = self.generate_random_zones()
        self.generate_resources_costs(Z)
        E2 = self.generate_optional_equipment(Z)
        res = {'Z': Z, 'E2': E2}

        max_employee_effort = {node: np.random.lognormal(mean=7, sigma=1.0) for node in Z.nodes}
        volunteer_probabilities = {(u, v): np.random.uniform(0.3, 0.9) for u, v in Z.edges}
        zone_time_windows = {node: (np.random.randint(1, 100), np.random.randint(100, 200)) for node in Z.nodes}
        equipment_costs = {(u, v): np.random.uniform(1.0, 50.0) for u, v in Z.edges}
        
        res.update({
            'max_employee_effort': max_employee_effort,
            'volunteer_probabilities': volunteer_probabilities,
            'zone_time_windows': zone_time_windows,
            'equipment_costs': equipment_costs,
        })
        return res
    
    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        Z, E2 = instance['Z'], instance['E2']
        
        max_employee_effort = instance['max_employee_effort']
        volunteer_probabilities = instance['volunteer_probabilities']
        zone_time_windows = instance['zone_time_windows']
        equipment_costs = instance['equipment_costs']
        
        model = Model("EPLO")
        zone_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in Z.nodes}
        equipment_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in Z.edges}
        
        employee_effort_vars = {node: model.addVar(vtype="C", name=f"EmployeeEffort_{node}") for node in Z.nodes}
        volunteer_scheduling_vars = {(u, v): model.addVar(vtype="B", name=f"VolunteerScheduling_{u}_{v}") for u, v in Z.edges}
        zone_capacity_vars = {node: model.addVar(vtype="I", name=f"ZoneCapacity_{node}") for node in Z.nodes}
        
        objective_expr = quicksum(
            Z.nodes[node]['importance'] * zone_vars[f"x{node}"]
            for node in Z.nodes
        )

        objective_expr -= quicksum(
            Z[u][v]['equipment_cost'] * equipment_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        for u, v in Z.edges:
            if (u, v) in E2:
                model.addCons(
                    zone_vars[f"x{u}"] + zone_vars[f"x{v}"] - equipment_vars[f"y{u}_{v}"] <= 1,
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
                zone_time_windows[node][0] <= zone_capacity_vars[node],
                name=f"ZoneTimeWindowStart_{node}"
            )
            model.addCons(
                zone_capacity_vars[node] <= zone_time_windows[node][1],
                name=f"ZoneTimeWindowEnd_{node}"
            )

        ### new constraints and variables and objective code ends here
        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_z': 100,
        'max_z': 130,
        'er_prob': 0.66,
        'equipment_param': 500.0,
        'alpha': 0.77,
        'facility_min_count': 15,
        'facility_max_count': 25,
    }

    eplo = EPLO(parameters, seed=seed)
    instance = eplo.generate_instance()
    solve_status, solve_time = eplo.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")