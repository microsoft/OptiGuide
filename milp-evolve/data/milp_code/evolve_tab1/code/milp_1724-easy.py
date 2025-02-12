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
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
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
        
        res = {
            'commodities': commodities,
            'adj_mat': adj_mat,
            'edge_list': edge_list,
            'incommings': incommings,
            'outcommings': outcommings,
            'facility_costs': facility_costs,
            'region_benefits': region_benefits,
            'capacities': capacities,
            'demands': demands
        }

        # New instance data
        node_capacities = np.random.randint(self.min_node_capacity, self.max_node_capacity + 1, self.n_nodes)
        traversal_costs = np.random.uniform(1, 10, (self.n_nodes, self.n_nodes))

        res.update({
            'node_capacities': node_capacities,
            'traversal_costs': traversal_costs
        })
        
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
        node_capacities = instance['node_capacities']
        traversal_costs = instance['traversal_costs']
        
        model = Model("FCMCNF")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        
        open_vars = {i: model.addVar(vtype="B", name=f"Facility_{i+1}") for i in range(self.n_nodes)}
        assign_vars = {(i, j): model.addVar(vtype="C", name=f"Assign_{i+1}_{j+1}") for i in range(self.n_nodes) for j in range(self.n_nodes)}

        # Symmetry Breaking: s variables
        s_vars = {(i, j): model.addVar(vtype="B", name=f"s_{i+1}_{j+1}") for i in range(self.n_nodes) for j in range(self.n_nodes)}

        # New Variables for Hierarchical Flow Control
        t_vars = {(i, j, k): model.addVar(vtype="C", name=f"t_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}

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

        # Add variable flow costs based on traversal costs
        objective_expr += quicksum(
            traversal_costs[i, j] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )

        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        for i in range(self.n_nodes):
            model.addCons(quicksum(assign_vars[(i, j)] for j in range(self.n_nodes)) <= capacities[i], f"Facility_{i+1}_Capacity")
            for j in range(self.n_nodes):
                model.addCons(assign_vars[(i, j)] <= capacities[i] * s_vars[(i, j)], f"ConvexHull_1_{i+1}_{j+1}")
                model.addCons(open_vars[i] >= s_vars[(i, j)], f"ConvexHull_2_{i+1}_{j+1}")
            model.addCons(open_vars[i] <= quicksum(s_vars[(i, l)] for l in range(self.n_nodes)), f"ConvexHull_3_{i+1}")

        for j in range(self.n_nodes):
            model.addCons(quicksum(assign_vars[(i, j)] for i in range(self.n_nodes)) == demands[j], f"Demand_{j+1}")

        # Symmetry breaking constraints
        for i in range(1, self.n_nodes):
            model.addCons(open_vars[i] <= open_vars[i-1], f"Symmetry_{i+1}")

        # New Hierarchical Flow Control Constraints
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                for k in range(self.n_commodities):
                    if (i, j) in edge_list:
                        model.addCons(t_vars[(i, j, k)] <= x_vars[f"x_{i+1}_{j+1}_{k+1}"], f"Hierarchical_1_{i+1}_{j+1}_{k+1}")
                        model.addCons(t_vars[(i, j, k)] >= x_vars[f"x_{i+1}_{j+1}_{k+1}"] - (1 - y_vars[f"y_{i+1}_{j+1}"]), f"Hierarchical_2_{i+1}_{j+1}_{k+1}")

        # Node Capacity Constraints
        for j in range(self.n_nodes):
            model.addCons(quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for i in incommings[j] for k in range(self.n_commodities)) <= node_capacities[j], f"NodeCap_{j+1}")

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 15,
        'max_n_nodes': 22,
        'min_n_commodities': 15,
        'max_n_commodities': 45,
        'c_range': (8, 37),
        'd_range': (90, 900),
        'ratio': 500,
        'k_max': 10,
        'er_prob': 0.59,
        'min_facility_cost': 300,
        'max_facility_cost': 750,
        'min_region_benefit': 20,
        'max_region_benefit': 80,
        'min_facility_cap': 25,
        'max_facility_cap': 50,
        'min_node_capacity': 100,
        'max_node_capacity': 500,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")