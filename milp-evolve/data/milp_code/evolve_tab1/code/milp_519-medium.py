import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class WLPSC:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    # Data Generation
    def generate_location_graph(self):
        # Generate a random geographical network with fixed warehouse and customer nodes
        G = nx.random_geometric_graph(self.n_total_nodes, self.geo_radius, seed=self.seed)
        adj_mat = np.zeros((self.n_total_nodes, self.n_total_nodes), dtype=object)
        edge_list = []
        occupancy_list = [random.randint(1, self.max_capacity) for _ in range(self.n_warehouses)]
        demand_list = [random.randint(1, self.max_demand) for _ in range(self.n_customers)]

        for i, j in G.edges:
            cost = np.random.uniform(*self.cost_range)
            adj_mat[i, j] = cost
            edge_list.append((i, j))

        warehouses = range(self.n_warehouses)
        customers = range(self.n_customers, self.n_total_nodes)
        
        return G, adj_mat, edge_list, occupancy_list, demand_list, warehouses, customers

    def generate_instance(self):
        self.n_total_nodes = self.n_warehouses + self.n_customers
        G, adj_mat, edge_list, occupancy_list, demand_list, warehouses, customers = self.generate_location_graph()

        res = {
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'occupancy_list': occupancy_list, 
            'demand_list': demand_list, 
            'warehouses': warehouses, 
            'customers': customers
        }

        crew_availability = np.random.randint(1, self.max_crew_availability, self.n_warehouses)
        emission_limits = np.random.uniform(*self.emission_limits_range, len(edge_list))
        
        res.update({
            'crew_availability': crew_availability,
            'emission_limits': emission_limits
        })
        
        return res

    # PySCIPOpt Modeling
    def solve(self, instance):
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        occupancy_list = instance['occupancy_list']
        demand_list = instance['demand_list']
        warehouses = instance['warehouses']
        customers = instance['customers']
        crew_availability = instance['crew_availability']
        emission_limits = instance['emission_limits']
        
        model = Model("WLPSC")
        y_vars = {f"F_open_{w+1}": model.addVar(vtype="B", name=f"F_open_{w+1}") for w in warehouses}
        x_vars = {f"A_assign_{w+1}_{c+1}": model.addVar(vtype="B", name=f"A_assign_{w+1}_{c+1}") for w in warehouses for c in customers}
        u_vars = {f"T_capacity_{w+1}": model.addVar(vtype="I", name=f"T_capacity_{w+1}") for w in warehouses}
        
        crew_vars = {f"crew_{w+1}": model.addVar(vtype="I", name=f"crew_{w+1}") for w in warehouses}
        emission_vars = {f"emission_{i+1}": model.addVar(vtype="C", name=f"emission_{i+1}") for i in range(len(edge_list))}

        # Objective function: minimize the total shipping cost and crew costs
        objective_expr = quicksum(
            adj_mat[w, c] * x_vars[f"A_assign_{w+1}_{c+1}"]
            for w in warehouses for c in customers
        )
        # Adding fixed cost for opening a warehouse
        objective_expr += quicksum(
            self.fixed_cost * y_vars[f"F_open_{w+1}"]
            for w in warehouses
        )
        # Adding crew costs
        objective_expr += quicksum(
            self.crew_cost_per_hour * crew_vars[f"crew_{w+1}"]
            for w in warehouses
        )

        model.setObjective(objective_expr, "minimize")

        # Constraints
        # Each customer is assigned to exactly one warehouse
        for c in customers:
            model.addCons(quicksum(x_vars[f"A_assign_{w+1}_{c+1}"] for w in warehouses) == 1, f"Assign_{c+1}")

        # Warehouse should be open if it accepts any customers
        for w in warehouses:
            for c in customers:
                model.addCons(x_vars[f"A_assign_{w+1}_{c+1}"] <= y_vars[f"F_open_{w+1}"], f"Open_Cond_{w+1}_{c+1}")

        # Warehouse capacity constraint
        for w in warehouses:
            model.addCons(quicksum(demand_list[c-self.n_customers] * x_vars[f"A_assign_{w+1}_{c+1}"] for c in customers) <= occupancy_list[w], f"Capacity_{w+1}")
        
        # Crew availability constraints
        for w in warehouses:
            model.addCons(crew_vars[f"crew_{w+1}"] <= crew_availability[w], f"Crew_Availability_{w+1}")
        
        # Emission constraints
        for idx, (i, j) in enumerate(edge_list):
            model.addCons(emission_vars[f"emission_{idx+1}"] <= emission_limits[idx], f"Emission_Limits_{idx+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_warehouses': 30,
        'n_customers': 787,
        'cost_range': (300, 1200),
        'max_capacity': 1000,
        'max_demand': 700,
        'fixed_cost': 5000,
        'geo_radius': 0.45,
        'max_crew_availability': 40,
        'crew_cost_per_hour': 25,
        'emission_limits_range': (500, 5000),
    }

    wlps = WLPSC(parameters, seed=seed)
    instance = wlps.generate_instance()
    solve_status, solve_time = wlps.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")