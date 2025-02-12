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

        return {
            'G': G,
            'max_task_effort': max_task_effort,
            'project_benefits': project_benefits,
            'task_probabilities': task_probabilities,
            'critical_tasks': critical_tasks,
            'employee_efficiencies': employee_efficiencies,
            'resource_availability': resource_availability
        }

    def solve(self, instance):
        G = instance['G']
        max_task_effort = instance['max_task_effort']
        project_benefits = instance['project_benefits']
        task_probabilities = instance['task_probabilities']
        critical_tasks = instance['critical_tasks']
        employee_efficiencies = instance['employee_efficiencies']
        resource_availability = instance['resource_availability']

        model = Model("WorkforceAllocationOptimization")

        employee_vars = {node: model.addVar(vtype="C", name=f"Employee_{node}") for node in G.nodes}
        task_vars = {(u, v): model.addVar(vtype="B", name=f"Task_{u}_{v}") for u, v in G.edges}
        critical_task_vars = {node: model.addVar(vtype="B", name=f"Critical_{node}") for node in G.nodes}
        resource_vars = {node: model.addVar(vtype="B", name=f"Resource_{node}") for node in G.nodes}

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

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_employees': 27,
        'max_employees': 450,
        'task_prob': 0.24,
        'min_complexity': 135,
        'max_complexity': 150,
        'min_proficiency': 0,
        'max_proficiency': 7,
        'min_task_effort': 105,
        'max_task_effort': 500,
        'min_benefit': 0.8,
        'max_benefit': 45.0,
        'max_task_size': 500,
        'min_efficiency': 0,
        'max_efficiency': 700,
    }
    
    optimizer = WorkforceAllocationOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")