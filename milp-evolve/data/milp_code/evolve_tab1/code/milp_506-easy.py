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
            # Introducing stochasticity to task complexity with normal distribution
            G[u][v]['complexity'] = np.random.normal(self.avg_complexity, self.std_complexity)
        return G

    def generate_employee_data(self, G):
        for node in G.nodes:
            # Introducing stochasticity to employee proficiency with normal distribution
            G.nodes[node]['proficiency'] = np.random.normal(self.avg_proficiency, self.std_proficiency)
        return G

    def get_instance(self):
        G = self.generate_skill_graph()
        G = self.generate_task_complexity(G)
        G = self.generate_employee_data(G)

        max_task_effort = {node: np.random.randint(self.min_task_effort, self.max_task_effort) for node in G.nodes}
        project_benefits = {node: np.random.uniform(self.min_benefit, self.max_benefit) for node in G.nodes}
        task_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        critical_tasks = [set(task) for task in nx.find_cliques(G.to_undirected()) if len(task) <= self.max_task_size]
        employee_efficiencies = {node: np.random.normal(self.avg_efficiency, self.std_efficiency) for node in G.nodes}
        resource_availability = {node: np.random.randint(0, 2) for node in G.nodes}

        proficiency_groups = {i: set(random.sample(G.nodes, int(len(G.nodes) / self.num_proficiency_groups))) for i in range(self.num_proficiency_groups)}

        task_difficulty = {edge: np.random.normal(self.avg_task_difficulty, self.std_task_difficulty) for edge in G.edges}
        employee_capacity = {node: np.random.randint(5, 20) for node in G.nodes}
        resource_limits = {node: np.random.randint(10, 50) for node in G.nodes}

        return {
            'G': G,
            'max_task_effort': max_task_effort,
            'project_benefits': project_benefits,
            'task_probabilities': task_probabilities,
            'critical_tasks': critical_tasks,
            'employee_efficiencies': employee_efficiencies,
            'resource_availability': resource_availability,
            'proficiency_groups': proficiency_groups,
            'task_difficulty': task_difficulty,
            'employee_capacity': employee_capacity,
            'resource_limits': resource_limits
        }

    def solve(self, instance):
        G = instance['G']
        max_task_effort = instance['max_task_effort']
        project_benefits = instance['project_benefits']
        task_probabilities = instance['task_probabilities']
        critical_tasks = instance['critical_tasks']
        employee_efficiencies = instance['employee_efficiencies']
        resource_availability = instance['resource_availability']
        proficiency_groups = instance['proficiency_groups']
        task_difficulty = instance['task_difficulty']
        employee_capacity = instance['employee_capacity']
        resource_limits = instance['resource_limits']

        model = Model("WorkforceAllocationOptimization")

        employee_vars = {node: model.addVar(vtype="C", name=f"Employee_{node}") for node in G.nodes}
        task_vars = {(u, v): model.addVar(vtype="B", name=f"Task_{u}_{v}") for u, v in G.edges}
        critical_task_vars = {node: model.addVar(vtype="B", name=f"Critical_{node}") for node in G.nodes}
        resource_vars = {node: model.addVar(vtype="B", name=f"Resource_{node}") for node in G.nodes}

        group_vars = {(node, group): model.addVar(vtype="B", name=f"Group_{group}_{node}") 
                      for node in G.nodes for group in proficiency_groups}
        
        work_assignment_vars = {(u, v): model.addVar(vtype="B", name=f"Work_Assignment_{u}_{v}") for u, v in G.edges}
        resource_usage_vars = {node: model.addVar(vtype="C", lb=0, name=f"Resource_Usage_{node}") for node in G.nodes}
        
        M = self.Big_M  # Big M constant

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

        for group, members in proficiency_groups.items():
            for task in critical_tasks:
                model.addCons(
                    quicksum(group_vars[(node, group)] for node in task if node in members) <= 1,
                    name=f"SetPackingGroup_{group}_Task_{task}"
                )

        for node in G.nodes:
            model.addCons(
                quicksum(task_vars[(u,v)] for u,v in G.edges if u == node or v == node) <= resource_availability[node] * quicksum(task_probabilities[(u,v)] for u,v in G.edges),
                name=f"KnapsackResourceUtilization_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                quicksum(group_vars[(u, group)] for group in proficiency_groups if u in proficiency_groups[group]) >= task_vars[(u, v)],
                name=f"TaskCoverage_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                resource_usage_vars[node] <= M * resource_vars[node],
                name=f"ResourceUsageCondition_{node}"
            )
            model.addCons(
                resource_usage_vars[node] >= employee_capacity[node] * resource_vars[node],
                name=f"ResourceLowerBoundCondition_{node}"
            )
            model.addCons(
                resource_usage_vars[node] <= resource_limits[node],
                name=f"ResourceLimit_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                work_assignment_vars[(u, v)] <= task_vars[(u, v)] * M,
                name=f"WorkAssignmentCondition_{u}_{v}"
            )
            for w in G.nodes:
                if w != u and (u, w) in G.edges:
                    model.addCons(
                        work_assignment_vars[(u, w)] + work_assignment_vars[(u, v)] <= 1,
                        name=f"EmployeeCapacityLimit_{u}"
                    )

        # New constraints using Big M:
        for u, v in G.edges:
            model.addCons(
                G[u][v]['complexity'] - G.nodes[u]['proficiency'] <= M * (1 - task_vars[(u, v)]),
                name=f"ComplexityCondition_{u}_{v}"
            )

            for g in proficiency_groups:
                if u in proficiency_groups[g]:
                    model.addCons(
                        task_vars[(u, v)] <= group_vars[(u, g)],
                        name=f"GroupTaskDependency_{u}_{g}"
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
        'max_employees': 505,
        'task_prob': 0.1,
        'avg_complexity': 391,
        'std_complexity': 200,
        'avg_proficiency': 87.5,
        'std_proficiency': 280,
        'min_task_effort': 945,
        'max_task_effort': 3000,
        'min_benefit': 0.24,
        'max_benefit': 180.0,
        'max_task_size': 750,
        'avg_efficiency': 1050,
        'std_efficiency': 375,
        'num_proficiency_groups': 450,
        'Big_M': 3000,
        'avg_task_difficulty': 35,
        'std_task_difficulty': 14,
    }

    optimizer = WorkforceAllocationOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")