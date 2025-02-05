import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from itertools import combinations

class GlobalSupplyChainProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        centers_graph = nx.barabasi_albert_graph(self.number_of_centers, 5)
        setup_cost = np.random.randint(self.min_setup_cost, self.max_setup_cost, self.number_of_centers)
        delivery_costs = np.random.randint(self.min_delivery_cost, self.max_delivery_cost, (self.number_of_centers, self.number_of_customers))
        resilience_impact = {i: np.random.uniform(0.8, 1.2) for i in range(self.number_of_centers)}
        environmental_penalty = np.random.randint(self.min_env_penalty, self.max_env_penalty, self.number_of_centers)
        regulation_cost = np.random.randint(self.min_regulation_cost, self.max_regulation_cost, self.number_of_centers)
        
        labor_availability = np.random.uniform(0.4, 1.0, self.number_of_centers)
        disaster_probabilities = np.random.uniform(0.01, 0.05, (4, self.number_of_centers))  # for 4 types of disruptions: natural, geopolitical, pandemic, others

        delivery_feasibility = np.random.randint(2, size=(self.number_of_centers, self.number_of_customers))
        transportation_caps = np.random.randint(self.min_transport_capacity, self.max_transport_capacity, (self.number_of_centers, self.number_of_customers))

        res = {
            'centers_graph': centers_graph,
            'setup_cost': setup_cost,
            'delivery_costs': delivery_costs,
            'resilience_impact': resilience_impact,
            'environmental_penalty': environmental_penalty,
            'regulation_cost': regulation_cost,
            'labor_availability': labor_availability,
            'disaster_probabilities': disaster_probabilities,
            'delivery_feasibility': delivery_feasibility,
            'transportation_caps': transportation_caps
        }

        return res

    def solve(self, instance):
        centers_graph = instance['centers_graph']
        setup_cost = instance['setup_cost']
        delivery_costs = instance['delivery_costs']
        resilience_impact = instance['resilience_impact']
        environmental_penalty = instance['environmental_penalty']
        regulation_cost = instance['regulation_cost']
        labor_availability = instance['labor_availability']
        disaster_probabilities = instance['disaster_probabilities']
        delivery_feasibility = instance['delivery_feasibility']
        transportation_caps = instance['transportation_caps']

        number_of_centers = len(setup_cost)
        number_of_customers = delivery_costs.shape[1]
        number_of_scenarios = len(disaster_probabilities)

        model = Model("GlobalSupplyChainProblem")

        center_setup = {i: model.addVar(vtype="B", name=f"center_setup_{i}") for i in range(number_of_centers)}
        delivery_assignment = {(i, j): model.addVar(vtype="B", name=f"delivery_assignment_{i}_{j}") for i in range(number_of_centers) for j in range(number_of_customers)}
        operational_status = {(i, s): model.addVar(vtype="B", name=f"operational_status_{i}_{s}") for i in range(number_of_centers) for s in range(number_of_scenarios)}
        pollution_vars = {(i, j): model.addVar(vtype="C", name=f"pollution_{i}_{j}") for i in range(number_of_centers) for j in range(number_of_customers)}
        compliance_vars = {i: model.addVar(vtype="C", name=f"compliance_{i}") for i in range(number_of_centers)}

        objective_expr = quicksum(setup_cost[i] * center_setup[i] for i in range(number_of_centers))
        objective_expr += quicksum(delivery_costs[i][j] * delivery_assignment[(i, j)] for i in range(number_of_centers) for j in range(number_of_customers))
        objective_expr += quicksum(environmental_penalty[i] * pollution_vars[(i, j)] for i in range(number_of_centers) for j in range(number_of_customers))
        objective_expr += quicksum(regulation_cost[i] * compliance_vars[i] for i in range(number_of_centers))

        # Maximizing the resilience as part of the objective
        resilience_term = quicksum(resilience_impact[i] * operational_status[(i, s)] for i in range(number_of_centers) for s in range(number_of_scenarios))
        objective_expr -= resilience_term

        model.setObjective(objective_expr, "minimize")

        for j in range(number_of_customers):
            model.addCons(quicksum(center_setup[i] * delivery_feasibility[i][j] for i in range(number_of_centers)) >= 1, f"CustomerCoverage_{j}")

        for i in range(number_of_centers):
            for j in range(number_of_customers):
                model.addCons(delivery_assignment[(i, j)] <= transportation_caps[i][j], f"TransportCapacity_{i}_{j}")

        for i in range(number_of_centers):
            for s in range(number_of_scenarios):
                model.addCons(operational_status[(i, s)] <= labor_availability[i] / disaster_probabilities[s][i], f"LaborAvailability_{i}_{s}")

        for i in range(number_of_centers):
            model.addCons(compliance_vars[i] == quicksum(delivery_assignment[(i, j)] for j in range(number_of_customers)), f"Compliance_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_centers': 1750,
        'number_of_customers': 100,
        'min_setup_cost': 5000,
        'max_setup_cost': 10000,
        'min_delivery_cost': 500,
        'max_delivery_cost': 2000,
        'min_env_penalty': 2000,
        'max_env_penalty': 3000,
        'min_regulation_cost': 900,
        'max_regulation_cost': 1500,
        'min_transport_capacity': 1600,
        'max_transport_capacity': 2000,
    }

    supply_chain_optimization = GlobalSupplyChainProblem(parameters, seed=seed)
    instance = supply_chain_optimization.generate_instance()
    solve_status, solve_time = supply_chain_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")