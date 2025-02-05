import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, probability, seed=None):
        graph = nx.erdos_renyi_graph(number_of_nodes, probability, seed=seed)
        edges = set(graph.edges)
        degrees = [d for (n, d) in graph.degree]
        neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes}
        return Graph(number_of_nodes, edges, degrees, neighbors)

class HealthcareResourceOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        personnel_costs = np.random.randint(self.min_personnel_cost, self.max_personnel_cost + 1, self.n_departments)
        supply_costs = np.random.randint(self.min_supply_cost, self.max_supply_cost + 1, (self.n_departments, self.n_supplies))
        capacities = np.random.randint(self.min_healthcare_capacity, self.max_healthcare_capacity + 1, self.n_departments)
        supplies = np.random.gamma(2, 2, self.n_supplies).astype(int) + 1

        graph = Graph.erdos_renyi(self.n_departments, self.link_probability, seed=self.seed)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))

        transportation_costs = np.random.randint(1, 20, (self.n_departments, self.n_departments))
        transportation_capacities = np.random.randint(self.transportation_capacity_min, self.transportation_capacity_max + 1, (self.n_departments, self.n_departments))

        nutritional_impact_costs = np.random.randint(10, 50, (self.n_departments, self.n_departments))
        efficiency_costs = np.random.randint(5, 30, (self.n_departments, self.n_departments))
        personnel_penalties = np.random.uniform(low=0.5, high=2.0, size=(self.n_departments, self.n_departments))

        environmental_impact = np.random.uniform(0.1, 10, (self.n_departments, self.n_departments))
        penalty_costs = np.random.uniform(10, 50, self.n_departments)
        
        financial_rewards = np.random.uniform(10, 100, self.n_supplies)
        energy_cost = np.random.uniform(0.1, 1)

        return {
            "personnel_costs": personnel_costs,
            "supply_costs": supply_costs,
            "capacities": capacities,
            "supplies": supplies,
            "graph": graph,
            "edge_weights": edge_weights,
            "transportation_costs": transportation_costs,
            "transportation_capacities": transportation_capacities,
            "nutritional_impact_costs": nutritional_impact_costs,
            "efficiency_costs": efficiency_costs,
            "personnel_penalties": personnel_penalties,
            "environmental_impact": environmental_impact,
            "penalty_costs": penalty_costs,
            "financial_rewards": financial_rewards,
            "energy_cost": energy_cost,
        }

    def solve(self, instance):
        personnel_costs = instance['personnel_costs']
        supply_costs = instance['supply_costs']
        capacities = instance['capacities']
        supplies = instance['supplies']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        transportation_costs = instance['transportation_costs']
        transportation_capacities = instance['transportation_capacities']
        nutritional_impact_costs = instance['nutritional_impact_costs']
        efficiency_costs = instance['efficiency_costs']
        personnel_penalties = instance['personnel_penalties']
        environmental_impact = instance['environmental_impact']
        penalty_costs = instance['penalty_costs']
        financial_rewards = instance['financial_rewards']
        energy_cost = instance['energy_cost']

        model = Model("HealthcareResourceOptimization")
        n_departments = len(personnel_costs)
        n_supplies = len(supply_costs[0])

        new_personnel = {c: model.addVar(vtype="B", name=f"NewPersonnel_Allocation_{c}") for c in range(n_departments)}
        medical_supplies = {(c, s): model.addVar(vtype="B", name=f"Department_{c}_Supply_{s}") for c in range(n_departments) for s in range(n_supplies)}
        resource_usage_vars = {(i, j): model.addVar(vtype="I", name=f"Resource_Usage_{i}_{j}") for i in range(n_departments) for j in range(n_departments)}
        new_transportation_vars = {(i, j): model.addVar(vtype="I", name=f"Transport_{i}_{j}") for i in range(n_departments) for j in range(n_departments)}
        
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_departments)}
        facility_energy_vars = {f: model.addVar(vtype="C", name=f"Energy_{f}") for f in range(n_departments)}

        model.setObjective(
            quicksum(personnel_costs[c] * new_personnel[c] for c in range(n_departments)) +
            quicksum(supply_costs[c, s] * medical_supplies[c, s] for c in range(n_departments) for s in range(n_supplies)) +
            quicksum(transportation_costs[i, j] * new_transportation_vars[i, j] for i in range(n_departments) for j in range(n_departments)) +
            quicksum(nutritional_impact_costs[i, j] * new_transportation_vars[i, j] for i in range(n_departments) for j in range(n_departments)) +
            quicksum(efficiency_costs[i, j] * new_transportation_vars[i, j] for i in range(n_departments) for j in range(n_departments)) +
            quicksum(personnel_penalties[i, j] * new_transportation_vars[i, j] for i in range(n_departments) for j in range(n_departments)) +
            quicksum(environmental_impact[i, j] * new_transportation_vars[i, j] for i in range(n_departments) for j in range(n_departments)) +
            quicksum(penalty_costs[i] for i in range(n_departments)) -
            quicksum(financial_rewards[s] * medical_supplies[c, s] for c in range(n_departments) for s in range(n_supplies)) +
            quicksum(energy_cost * facility_energy_vars[c] for c in range(n_departments)),
            "minimize"
        )

        for s in range(n_supplies):
            model.addCons(quicksum(medical_supplies[c, s] for c in range(n_departments)) == 1, f"Supply_{s}_Allocation")
    
        for c in range(n_departments):
            for s in range(n_supplies):
                model.addCons(medical_supplies[c, s] <= new_personnel[c], f"Department_{c}_Serve_{s}")
    
        for c in range(n_departments):
            model.addCons(quicksum(supplies[s] * medical_supplies[c, s] for s in range(n_supplies)) <= capacities[c], f"Department_{c}_Capacity")

        for edge in graph.edges:
            model.addCons(new_personnel[edge[0]] + new_personnel[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for i in range(n_departments):
            model.addCons(
                quicksum(resource_usage_vars[i, j] for j in range(n_departments) if i != j) ==
                quicksum(resource_usage_vars[j, i] for j in range(n_departments) if i != j),
                f"Usage_Conservation_{i}"
            )
    
        for j in range(n_departments):
            model.addCons(
                quicksum(new_transportation_vars[i, j] for i in range(n_departments) if i != j) ==
                quicksum(medical_supplies[j, s] for s in range(n_supplies)),
                f"Transport_Conservation_{j}"
            )
    
        for i in range(n_departments):
            for j in range(n_departments):
                if i != j:
                    model.addCons(new_transportation_vars[i, j] <= transportation_capacities[i, j], f"Transport_Capacity_{i}_{j}")
    
        for c in range(n_departments):
            model.addCons(
                quicksum(medical_supplies[c, s] for s in range(n_supplies)) <= n_supplies * new_personnel[c],
                f"Convex_Hull_{c}"
            )

        for c in range(n_departments):
            model.addCons(
                facility_energy_vars[c] == quicksum(medical_supplies[c, s] * 150 for s in range(n_supplies)),
                f"Energy_Consumption_{c}"
            )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_departments': 18,
        'n_supplies': 504,
        'min_supply_cost': 2478,
        'max_supply_cost': 3000,
        'min_personnel_cost': 1397,
        'max_personnel_cost': 5000,
        'min_healthcare_capacity': 1180,
        'max_healthcare_capacity': 2154,
        'link_probability': 0.1,
        'transportation_capacity_min': 1170,
        'transportation_capacity_max': 3000,
    }
    resource_optimizer = HealthcareResourceOptimization(parameters, seed=seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")