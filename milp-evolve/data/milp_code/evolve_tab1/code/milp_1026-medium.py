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
            # integer demands
            demand_k = int(np.random.uniform(*self.d_range))
            commodities.append((o_k, d_k, demand_k))
        return commodities

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        # Additional instance data for Data Center Placement Optimization
        num_data_centers = np.random.randint(self.min_data_centers, self.max_data_centers+1)
        node_connection_costs = np.random.randint(50, 300, size=(self.n_nodes, num_data_centers))
        operational_costs_type1 = np.random.randint(1000, 3000, size=num_data_centers)
        operational_costs_type2 = np.random.randint(2500, 5000, size=num_data_centers)
        server_capacity_type1 = np.random.randint(1000, 3000, size=num_data_centers)
        server_capacity_type2 = np.random.randint(3000, 5000, size=num_data_centers)
        distances = np.random.randint(1, 100, size=(self.n_nodes, num_data_centers))

        res = {
            'commodities': commodities, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings,
            'num_data_centers': num_data_centers,
            'node_connection_costs': node_connection_costs,
            'operational_costs_type1': operational_costs_type1,
            'operational_costs_type2': operational_costs_type2,
            'server_capacity_type1': server_capacity_type1,
            'server_capacity_type2': server_capacity_type2,
            'distances': distances
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        
        num_data_centers = instance['num_data_centers']
        node_connection_costs = instance['node_connection_costs']
        operational_costs_type1 = instance['operational_costs_type1']
        operational_costs_type2 = instance['operational_costs_type2']
        server_capacity_type1 = instance['server_capacity_type1']
        server_capacity_type2 = instance['server_capacity_type2']
        distances = instance['distances']

        model = Model("FCMCNF_Extended")

        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}

        mega_server_type1 = {j: model.addVar(vtype="B", name=f"mega_server_type1_{j}") for j in range(num_data_centers)}
        mega_server_type2 = {j: model.addVar(vtype="B", name=f"mega_server_type2_{j}") for j in range(num_data_centers)}
        node_connection = {(i, j): model.addVar(vtype="B", name=f"node_connection_{i}_{j}") for i in range(self.n_nodes) for j in range(num_data_centers)}

        # Extend the objective function 
        objective_expr = quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(self.n_commodities)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(node_connection[i, j] * node_connection_costs[i, j] for i in range(self.n_nodes) for j in range(num_data_centers))
        objective_expr += quicksum(mega_server_type1[j] * operational_costs_type1[j] for j in range(num_data_centers))
        objective_expr += quicksum(mega_server_type2[j] * operational_costs_type2[j] for j in range(num_data_centers))
        objective_expr += quicksum(node_connection[i, j] * distances[i, j] for i in range(self.n_nodes) for j in range(num_data_centers))

        # Flow conservation constraints
        for i in range(self.n_nodes):
            for k in range(self.n_commodities):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Capacity constraint on each arc
        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(self.n_commodities))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

        # Node connection constraints
        for i in range(self.n_nodes):
            model.addCons(quicksum(node_connection[i, j] for j in range(num_data_centers)) == 1, name=f"node_connection_{i}")

        for j in range(num_data_centers):
            for i in range(self.n_nodes):
                model.addCons(node_connection[i, j] <= mega_server_type1[j] + mega_server_type2[j], name=f"data_center_node_{i}_{j}")

        for j in range(num_data_centers):
            model.addCons(quicksum(commodities[k][2] * node_connection[commodities[k][0], j] for k in range(self.n_commodities))
                          <= mega_server_type1[j] * server_capacity_type1[j] + 
                             mega_server_type2[j] * server_capacity_type2[j], name=f"capacity_{j}")
        
        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 20,
        'max_n_nodes': 30,
        'min_n_commodities': 30,
        'max_n_commodities': 45, 
        'c_range': (11, 50),
        'd_range': (10, 100),
        'ratio': 100,
        'k_max': 10,
        'er_prob': 0.3,
        'min_data_centers': 3,
        'max_data_centers': 7,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")