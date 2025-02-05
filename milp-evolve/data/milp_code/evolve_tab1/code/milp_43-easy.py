import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WarehouseLocationProblem:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_graph(self, n):
        return nx.barabasi_albert_graph(n, self.ba_degree, seed=self.seed)

    def compute_distances(self, G, demand_centers, potential_warehouses):
        distances = {}
        for i in demand_centers:
            for j in potential_warehouses:
                try:
                    dist = nx.shortest_path_length(G, source=i, target=j, weight='weight')
                except nx.NetworkXNoPath:
                    dist = float('inf')
                distances[f"{i}_{j}"] = dist
        return distances

    def get_instance(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        G = self.generate_graph(n)
        num_demand_centers = n // 2
        num_potential_warehouses = n - num_demand_centers
        demand_centers = random.sample(range(n), num_demand_centers)
        potential_warehouses = list(set(range(n)) - set(demand_centers))

        distances = self.compute_distances(G, demand_centers, potential_warehouses)
        building_costs = {i: np.random.randint(1000, 10000) for i in potential_warehouses}
        capacity = {i: np.random.randint(20, 200) for i in potential_warehouses}
        demand = {i: np.random.randint(1, 20) for i in demand_centers}
        
        res = {
            'demand_centers': demand_centers,
            'potential_warehouses': potential_warehouses,
            'distances': distances,
            'building_costs': building_costs,
            'capacity': capacity,
            'demand': demand
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demand_centers = instance['demand_centers']
        potential_warehouses = instance['potential_warehouses']
        distances = instance['distances']
        building_costs = instance['building_costs']
        capacity = instance['capacity']
        demand = instance['demand']

        model = Model("WarehouseLocation")
        warehouse_vars = {i: model.addVar(vtype="B", name=f"wh_{i}") for i in potential_warehouses}
        assignment_vars = {f"{i}_{j}": model.addVar(vtype="I", name=f"assign_{i}_{j}")
                           for i in demand_centers for j in potential_warehouses}
        distance_penalty = model.addVar(vtype="C", name="distance_penalty")

        # Objective function: Minimize total costs (building + transportation) and incorporate distance penalties
        building_cost_expr = quicksum(building_costs[j] * warehouse_vars[j] for j in potential_warehouses)
        transportation_cost_expr = quicksum(distances[f"{i}_{j}"] * assignment_vars[f"{i}_{j}"]
                                            for i in demand_centers for j in potential_warehouses)
        total_cost_expr = building_cost_expr + transportation_cost_expr + distance_penalty
        model.setObjective(total_cost_expr, "minimize")

        # Constraint 1: Each demand center must be covered completely by the warehouses.
        for i in demand_centers:
            model.addCons(quicksum(assignment_vars[f"{i}_{j}"] for j in potential_warehouses) == demand[i],
                          name=f"coverage_{i}")

        # Constraint 2: Demand center i can only be assigned to an open warehouse j.
        for i in demand_centers:
            for j in potential_warehouses:
                model.addCons(assignment_vars[f"{i}_{j}"] <= demand[i] * warehouse_vars[j], name=f"assign_cond_{i}_{j}")

        # Constraint 3: No more than max_warehouses can be built.
        model.addCons(quicksum(warehouse_vars[j] for j in potential_warehouses) <= self.max_warehouses, 
                      name="max_warehouses")

        # Constraint 4: Respect the capacity of each warehouse.
        for j in potential_warehouses:
            model.addCons(quicksum(assignment_vars[f"{i}_{j}"] for i in demand_centers) <= capacity[j] * warehouse_vars[j],
                          name=f"capacity_{j}")
        
        # Constraint 5: Apply a distance penalty for far assignments.
        for i in demand_centers:
            for j in potential_warehouses:
                model.addCons(distance_penalty >= distances[f"{i}_{j}"] * assignment_vars[f"{i}_{j}"],
                              name=f"distance_penalty_{i}_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 150,
        'max_n': 400,
        'ba_degree': 50,
        'max_warehouses': 80,
    }

    warehouse_loc_problem = WarehouseLocationProblem(parameters, seed=seed)
    instance = warehouse_loc_problem.get_instance()
    solve_status, solve_time = warehouse_loc_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")