import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WorkforceAllocationOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_skill_graph(self):
        n_nodes = np.random.randint(self.min_employees, self.max_employees)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.task_prob, directed=True, seed=self.seed)
        return G

    def generate_task_complexity(self, G):
        for u, v in G.edges:
            G[u][v]['complexity'] = np.random.uniform(self.min_complexity, self.max_complexity)
        return G

    def generate_employee_data(self, G):
        for node in G.nodes:
            G.nodes[node]['proficiency'] = np.random.randint(self.min_proficiency, self.max_proficiency)
        return G

    def get_instance(self):
        G = self.generate_skill_graph()
        G = self.generate_task_complexity(G)
        G = self.generate_employee_data(G)

        max_task_effort = {node: np.random.randint(self.min_task_effort, self.max_task_effort) for node in G.nodes}
        project_benefits = {node: np.random.uniform(self.min_benefit, self.max_benefit) for node in G.nodes}
        task_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        critical_tasks = [set(task) for task in nx.find_cliques(G.to_undirected()) if len(task) <= self.max_task_size]
        employee_efficiencies = {node: np.random.randint(self.min_efficiency, self.max_efficiency) for node in G.nodes}
        resource_availability = {node: np.random.randint(0, 2) for node in G.nodes}

        # New Data for Extended Problem
        population_movements = {node: np.random.randint(self.min_movement, self.max_movement) for node in G.nodes}
        infrastructure_damage = {node: np.random.uniform(self.min_damage, self.max_damage) for node in G.nodes}
        fuel_consumptions = {(u, v): np.random.uniform(self.min_fuel, self.max_fuel) for u, v in G.edges}
        terrain_conditions = {(u, v): np.random.choice(['normal', 'flooded', 'landslide']) for u, v in G.edges}

        return {
            'G': G,
            'max_task_effort': max_task_effort,
            'project_benefits': project_benefits,
            'task_probabilities': task_probabilities,
            'critical_tasks': critical_tasks,
            'employee_efficiencies': employee_efficiencies,
            'resource_availability': resource_availability,
            'population_movements': population_movements,
            'infrastructure_damage': infrastructure_damage,
            'fuel_consumptions': fuel_consumptions,
            'terrain_conditions': terrain_conditions
        }

    def solve(self, instance):
        G = instance['G']
        max_task_effort = instance['max_task_effort']
        project_benefits = instance['project_benefits']
        task_probabilities = instance['task_probabilities']
        critical_tasks = instance['critical_tasks']
        employee_efficiencies = instance['employee_efficiencies']
        resource_availability = instance['resource_availability']
        population_movements = instance['population_movements']
        infrastructure_damage = instance['infrastructure_damage']
        fuel_consumptions = instance['fuel_consumptions']
        terrain_conditions = instance['terrain_conditions']

        model = Model("WorkforceAllocationOptimization")

        employee_vars = {node: model.addVar(vtype="C", name=f"Employee_{node}") for node in G.nodes}
        task_vars = {(u, v): model.addVar(vtype="B", name=f"Task_{u}_{v}") for u, v in G.edges}
        critical_task_vars = {node: model.addVar(vtype="B", name=f"Critical_{node}") for node in G.nodes}
        resource_vars = {node: model.addVar(vtype="B", name=f"Resource_{node}") for node in G.nodes}
        fuel_vars = {(u, v): model.addVar(vtype="C", name=f"Fuel_{u}_{v}") for u, v in G.edges}

        objective_expr = quicksum(
            project_benefits[node] * employee_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            G[u][v]['complexity'] * task_vars[(u, v)]
            for u, v in G.edges
        )
        objective_expr += quicksum(
            employee_efficiencies[node] * critical_task_vars[node]
            for node in G.nodes
        )
        objective_expr -= quicksum(
            resource_availability[node] * resource_vars[node]
            for node in G.nodes
        )

        for node in G.nodes:
            model.addCons(
                employee_vars[node] <= max_task_effort[node],
                name=f"MaxTaskEffort_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                task_vars[(u, v)] <= task_probabilities[(u, v)],
                name=f"TaskProbability_{u}_{v}"
            )
            model.addCons(
                task_vars[(u, v)] <= employee_vars[u],
                name=f"TaskAssignLimit_{u}_{v}"
            )
            # New fuel consumption constraint based on terrain conditions
            fuel_factor = 1 if terrain_conditions[(u, v)] == 'normal' else 1.5 if terrain_conditions[(u, v)] == 'flooded' else 2
            model.addCons(
                fuel_vars[(u, v)] == fuel_factor * fuel_consumptions[(u, v)] * task_vars[(u, v)],
                name=f"FuelConsumption_{u}_{v}"
            )
        
        # Adjusting constraints to use Set Packing for critical tasks
        for task in critical_tasks:
            model.addCons(
                quicksum(critical_task_vars[node] for node in task) <= 1,
                name=f"MaxOneCriticalTask_{task}"
            )

        for node in G.nodes:
            model.addCons(
                resource_vars[node] <= resource_availability[node],
                name=f"ResourceAvailability_{node}"
            )

        # New constraint to balance population movements and infrastructure damage
        for node in G.nodes:
            model.addCons(
                employee_vars[node] * population_movements[node] / (1 + infrastructure_damage[node]) <= max_task_effort[node],
                name=f"BalancedAllocation_{node}"
            )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_employees': 27,
        'max_employees': 225,
        'task_prob': 0.24,
        'min_complexity': 675,
        'max_complexity': 1050,
        'min_proficiency': 0,
        'max_proficiency': 63,
        'min_task_effort': 105,
        'max_task_effort': 1500,
        'min_benefit': 0.31,
        'max_benefit': 450.0,
        'max_task_size': 500,
        'min_efficiency': 0,
        'max_efficiency': 2100,
        'min_movement': 0,
        'max_movement': 5000,
        'min_damage': 0.45,
        'max_damage': 100.0,
        'min_fuel': 3.75,
        'max_fuel': 250.0,
    }
    
    optimizer = WorkforceAllocationOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")