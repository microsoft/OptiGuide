import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FCMCNF:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_nodes, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_nodes)}
        outcommings = {i: [] for i in range(self.n_nodes)}

        for i, j in G.edges:
            c_ij = np.random.uniform(*self.c_range)
            f_ij = np.random.uniform(self.c_range[0] * self.ratio, self.c_range[1] * self.ratio)
            u_ij = np.random.uniform(1, self.k_max + 1) * np.random.uniform(*self.d_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_commodities(self, G):
        commodities = []
        for k in range(self.n_commodities):
            while True:
                o_k = np.random.randint(0, self.n_nodes)
                d_k = np.random.randint(0, self.n_nodes)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        # Generate data for facility location
        self.n_facilities = self.n_nodes
        self.n_regions = self.n_nodes
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        region_benefits = np.random.randint(self.min_region_benefit, self.max_region_benefit + 1, (self.n_facilities, self.n_regions))
        capacities = np.random.randint(self.min_facility_cap, self.max_facility_cap + 1, self.n_facilities)
        demands = np.random.randint(1, 10, self.n_regions)

        # New data parameters from the second MILP
        total_budget = np.random.randint(50000, 100000)
        num_packages = 3
        package_cost = np.random.randint(500, 2000, size=num_packages)
        package_capacity = np.random.randint(50, 200, size=num_packages)
        max_packages_per_facility = np.random.randint(1, num_packages + 1, size=self.n_facilities)
        transportation_costs = np.random.randint(1, 20, size=(self.n_facilities, self.n_regions))
        seasonal_variation = np.random.normal(0, 0.1, size=self.n_facilities)  # New variability parameter

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'facility_costs': facility_costs,
            'region_benefits': region_benefits,
            'capacities': capacities,
            'demands': demands,
            'total_budget': total_budget,
            'num_packages': num_packages,
            'package_cost': package_cost,
            'package_capacity': package_capacity,
            'max_packages_per_facility': max_packages_per_facility,
            'transportation_costs': transportation_costs,
            'seasonal_variation': seasonal_variation,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        facility_costs = instance['facility_costs']
        region_benefits = instance['region_benefits']
        capacities = instance['capacities']
        demands = instance['demands']
        total_budget = instance['total_budget']
        num_packages = instance['num_packages']
        package_cost = instance['package_cost']
        package_capacity = instance['package_capacity']
        max_packages_per_facility = instance['max_packages_per_facility']
        transportation_costs = instance['transportation_costs']
        seasonal_variation = instance['seasonal_variation']
        
        model = Model("FCMCNF")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        
        open_vars = {i: model.addVar(vtype="B", name=f"Facility_{i+1}") for i in range(self.n_nodes)}
        assign_vars = {(i, j): model.addVar(vtype="C", name=f"Assign_{i+1}_{j+1}") for i in range(self.n_nodes) for j in range(self.n_nodes)}

        # No need for Big M variables under Convex Hull method
        z_vars = {(i, j, k): model.addVar(vtype="C", lb=0, ub=1, name=f"z_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}

        # New variables for packages and transportation flow
        PackageAvailable = {(i, k): model.addVar(vtype="B", name=f"PackageAvailable_{i+1}_{k+1}") for i in range(self.n_facilities) for k in range(num_packages)}
        PackageUsage = {(i, j, k): model.addVar(vtype="I", name=f"PackageUsage_{i+1}_{j+1}_{k+1}") for i in range(self.n_facilities) for j in range(self.n_regions) for k in range(num_packages)}
        TransportationFlow = {(i, j): model.addVar(vtype="C", name=f"TransportationFlow_{i+1}_{j+1}") for i in range(self.n_facilities) for j in range(self.n_regions)}

        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr -= quicksum(
            facility_costs[i] * open_vars[i]
            for i in range(self.n_nodes)
        )
        objective_expr += quicksum(
            region_benefits[i, j] * assign_vars[(i, j)]
            for i in range(self.n_nodes) for j in range(self.n_nodes)
        )

        # New objective terms
        objective_expr += quicksum(PackageAvailable[i, k] * package_cost[k] for i in range(self.n_facilities) for k in range(num_packages))
        objective_expr += quicksum(TransportationFlow[i, j] * transportation_costs[i][j] for i in range(self.n_facilities) for j in range(self.n_regions))

        # Flow Conservation constraints
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Arc Capacity constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")
            
            # Convex Hull Formulation constraints
            for k in range(self.n_commodities):
                model.addCons(x_vars[f"x_{i+1}_{j+1}_{k+1}"] <= z_vars[(i, j, k)] * adj_mat[i, j][2], f"convex_hull_x_{i+1}_{j+1}_{k+1}")
                model.addCons(z_vars[(i, j, k)] <= y_vars[f"y_{i+1}_{j+1}"], f"convex_hull_z_{i+1}_{j+1}_{k+1}")

        # Facility Assignment constraints
        for i in range(self.n_nodes):
            model.addCons(quicksum(assign_vars[(i, j)] for j in range(self.n_nodes)) <= capacities[i], f"Facility_{i+1}_Capacity")
             
        for j in range(self.n_nodes):
            model.addCons(quicksum(assign_vars[(i, j)] for i in range(self.n_nodes)) == demands[j], f"Demand_{j+1}")

        # Package constraints
        for i in range(self.n_facilities):
            model.addCons(quicksum(PackageAvailable[i, k] for k in range(num_packages)) <= max_packages_per_facility[i], name=f"package_limit_{i+1}")
        
        for i in range(self.n_facilities):
            for j in range(self.n_regions):
                model.addCons(quicksum(PackageUsage[i, j, k] * package_capacity[k] for k in range(num_packages)) >= demands[j], 
                              name=f"package_meet_demand_{i+1}_{j+1}")
                model.addCons(quicksum(PackageUsage[i, j, k] for k in range(num_packages)) <= quicksum(PackageAvailable[i, k] for k in range(num_packages)), 
                              name=f"package_usage_constraint_{i+1}_{j+1}")

        # Transportation flow constraints
        for i in range(self.n_facilities):
            slope = seasonal_variation[i]
            for j in range(self.n_regions):
                model.addCons(TransportationFlow[i, j] <= capacities[i] * (1 + slope) * open_vars[i], name=f"transportation_capacity_{i+1}_{j+1}")
                model.addCons(TransportationFlow[i, j] >= 0, name=f"positive_transportation_flow_{i+1}_{j+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 15,
        'max_n_nodes': 52,
        'min_n_commodities': 4,
        'max_n_commodities': 11,
        'c_range': (8, 37),
        'd_range': (70, 700),
        'ratio': 1800,
        'k_max': 500,
        'er_prob': 0.8,
        'min_facility_cost': 9,
        'max_facility_cost': 750,
        'min_region_benefit': 67,
        'max_region_benefit': 630,
        'min_facility_cap': 75,
        'max_facility_cap': 262,
        'total_budget': 75000,
        'num_packages': 9,
        'package_cost': (250, 750, 1000),
        'package_capacity': (450, 900, 1350),
        'max_packages_per_facility': (9, 18, 27),
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")