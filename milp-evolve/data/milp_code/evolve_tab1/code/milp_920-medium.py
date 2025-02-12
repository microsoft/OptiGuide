import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedTelecomWorkforceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ### Data generation for Telecom Network ###
    def generate_instance(self):
        assert self.n_hubs > 0 and self.n_neighborhoods > 0
        assert self.min_hub_cost >= 0 and self.max_hub_cost >= self.min_hub_cost
        assert self.min_connection_cost >= 0 and self.max_connection_cost >= self.min_connection_cost
        assert self.min_hub_capacity > 0 and self.max_hub_capacity >= self.min_hub_capacity
        
        hub_costs = np.random.randint(self.min_hub_cost, self.max_hub_cost + 1, self.n_hubs)
        connection_costs = np.random.randint(self.min_connection_cost, self.max_connection_cost + 1, (self.n_hubs, self.n_neighborhoods))
        capacities = np.random.randint(self.min_hub_capacity, self.max_hub_capacity + 1, self.n_hubs)

        ### Instance generation for workforce allocation ###
        G = self.generate_skill_graph()
        G = self.generate_task_complexity(G)
        G = self.generate_employee_data(G)

        max_task_effort = {node: np.random.randint(self.min_task_effort, self.max_task_effort) for node in G.nodes}
        project_benefits = {node: np.random.uniform(self.min_benefit, self.max_benefit) for node in G.nodes}
        task_probabilities = {(u, v): np.random.uniform(0.5, 1) for u, v in G.edges}
        critical_tasks = [set(task) for task in nx.find_cliques(G.to_undirected()) if len(task) <= self.max_task_size]
        employee_efficiencies = {node: np.random.randint(self.min_efficiency, self.max_efficiency) for node in G.nodes}

        return {
            "hub_costs": hub_costs,
            "connection_costs": connection_costs,
            "capacities": capacities,
            'G': G,
            'max_task_effort': max_task_effort,
            'project_benefits': project_benefits,
            'task_probabilities': task_probabilities,
            'critical_tasks': critical_tasks,
            'employee_efficiencies': employee_efficiencies,
        }

    ### Dataset generation: workforce ###
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

    ### PySCIPOpt modeling ###
    def solve(self, instance):
        hub_costs = instance['hub_costs']
        connection_costs = instance['connection_costs']
        capacities = instance['capacities']
        
        G = instance['G']
        max_task_effort = instance['max_task_effort']
        project_benefits = instance['project_benefits']
        task_probabilities = instance['task_probabilities']
        critical_tasks = instance['critical_tasks']
        employee_efficiencies = instance['employee_efficiencies']
        
        model = Model("SimplifiedTelecomWorkforceOptimization")
        n_hubs = len(hub_costs)
        n_neighborhoods = len(connection_costs[0])
        
        # Decision variables for telecom
        hub_vars = {h: model.addVar(vtype="B", name=f"Hub_{h}") for h in range(n_hubs)}
        connection_vars = {(h, n): model.addVar(vtype="B", name=f"Hub_{h}_Neighborhood_{n}") for h in range(n_hubs) for n in range(n_neighborhoods)}
        overflow = model.addVar(vtype="C", lb=0, name="Overflow")

        # Decision variables for workforce
        employee_vars = {node: model.addVar(vtype="C", name=f"Employee_{node}") for node in G.nodes}
        task_vars = {(u, v): model.addVar(vtype="B", name=f"Task_{u}_{v}") for u, v in G.edges}
        critical_task_vars = {node: model.addVar(vtype="B", name=f"Critical_{node}") for node in G.nodes}

        # Modified Objective: minimize total cost and maximize project benefits
        model.setObjective(
            quicksum(hub_costs[h] * hub_vars[h] for h in range(n_hubs)) +
            quicksum(connection_costs[h, n] * connection_vars[h, n] for h in range(n_hubs) for n in range(n_neighborhoods)) +
            1000 * overflow -
            quicksum(project_benefits[node] * employee_vars[node] for node in G.nodes),
            "minimize"
        )

        # Constraints: telecom
        for n in range(n_neighborhoods):
            model.addCons(quicksum(connection_vars[h, n] for h in range(n_hubs)) == 1, f"Neighborhood_{n}_Assignment")
        for h in range(n_hubs):
            for n in range(n_neighborhoods):
                model.addCons(connection_vars[h, n] <= hub_vars[h], f"Hub_{h}_Service_{n}")
        for h in range(n_hubs):
            model.addCons(quicksum(connection_vars[h, n] for n in range(n_neighborhoods)) <= capacities[h] + overflow, f"Hub_{h}_Capacity")

        # Constraints: workforce
        for node in G.nodes:
            model.addCons(employee_vars[node] <= max_task_effort[node], name=f"MaxTaskEffort_{node}")
        for u, v in G.edges:
            model.addCons(task_vars[(u, v)] <= task_probabilities[(u, v)], name=f"TaskProbability_{u}_{v}")
            model.addCons(task_vars[(u, v)] <= employee_vars[u], name=f"TaskAssignLimit_{u}_{v}")
        for task in critical_tasks:
            model.addCons(quicksum(critical_task_vars[node] for node in task) <= 1, name=f"MaxOneCriticalTask_{task}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hubs': 200,
        'n_neighborhoods': 50,
        'min_hub_cost': 3000,
        'max_hub_cost': 10000,
        'min_connection_cost': 900,
        'max_connection_cost': 1875,
        'min_hub_capacity': 1008,
        'max_hub_capacity': 3000,
        'min_employees': 70,
        'max_employees': 1000,
        'task_prob': 0.38,
        'min_complexity': 675,
        'max_complexity': 1500,
        'min_proficiency': 0,
        'max_proficiency': 21,
        'min_task_effort': 500,
        'max_task_effort': 900,
        'min_benefit': 10.0,
        'max_benefit': 140.0,
        'max_task_size': 45,
        'min_efficiency': 0,
        'max_efficiency': 400,
    }
    
    simplified_optimizer = SimplifiedTelecomWorkforceOptimization(parameters, seed=seed)
    instance = simplified_optimizer.generate_instance()
    solve_status, solve_time, objective_value = simplified_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")