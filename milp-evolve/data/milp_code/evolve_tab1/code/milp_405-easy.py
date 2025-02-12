import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class FacilityLocationTransportation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
        
    def generate_instance(self):
        # Randomly generate fixed costs for opening a facility
        fixed_costs = np.random.randint(self.min_cost, self.max_cost, self.number_of_facilities)

        # Randomly generate transportation costs between facilities and nodes
        transportation_costs = np.random.randint(self.min_cost, self.max_cost, (self.number_of_facilities, self.number_of_nodes))

        # Randomly generate capacities of facilities
        facility_capacities = np.random.randint(self.min_cap, self.max_cap, self.number_of_facilities)

        # Randomly generate demands for nodes
        node_demands = np.random.randint(self.min_demand, self.max_demand, self.number_of_nodes)
        
        # Randomly generate node-facility specific capacities
        node_facility_capacities = np.random.randint(self.node_facility_min_cap, self.node_facility_max_cap, (self.number_of_nodes, self.number_of_facilities))
        
        res = {
            'fixed_costs': fixed_costs,
            'transportation_costs': transportation_costs,
            'facility_capacities': facility_capacities,
            'node_demands': node_demands,
            'node_facility_capacities': node_facility_capacities,
        }
        
        ################## Incorporate FCMCNF Data Generation #################

        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes + 1)
        self.n_commodities = np.random.randint(self.min_n_commodities, self.max_n_commodities + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        commodities = self.generate_commodities(G)

        res['commodities'] = commodities
        res['adj_mat'] = adj_mat
        res['edge_list'] = edge_list
        res['incommings'] = incommings
        res['outcommings'] = outcommings

        return res

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

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        fixed_costs = instance['fixed_costs']
        transportation_costs = instance['transportation_costs']
        facility_capacities = instance['facility_capacities']
        node_demands = instance['node_demands']
        node_facility_capacities = instance['node_facility_capacities']
        commodities = instance['commodities']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        
        number_of_facilities = len(fixed_costs)
        number_of_nodes = len(node_demands)
        
        M = 1e6  # Big M constant

        model = Model("FacilityLocationTransportationMulticommodity")
        open_facility = {}
        transport_goods = {}
        node_demand_met = {}
        x_vars = {}
        y_vars = {}
        z_vars = {}

        # Decision variables: y[j] = 1 if facility j is open
        for j in range(number_of_facilities):
            open_facility[j] = model.addVar(vtype="B", name=f"y_{j}")

        # Decision variables: x[i][j] = amount of goods transported from facility j to node i
        for i in range(number_of_nodes):
            for j in range(number_of_facilities):
                transport_goods[(i, j)] = model.addVar(vtype="C", name=f"x_{i}_{j}")

        # Decision variables: z[i] = 1 if demand of node i is met
        for i in range(number_of_nodes):
            node_demand_met[i] = model.addVar(vtype="B", name=f"z_{i}")

        # New variables for multicommodity flows
        for (i, j) in edge_list:
            for k in range(len(commodities)):
                x_vars[f"x_{i+1}_{j+1}_{k+1}"] = model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}")
                z_vars[f"z_{i+1}_{j+1}_{k+1}"] = model.addVar(vtype="C", name=f"z_{i+1}_{j+1}_{k+1}")
            y_vars[f"y_{i+1}_{j+1}"] = model.addVar(vtype="B", name=f"y_{i+1}_{j+1}")
        
        # Objective: Minimize total cost
        objective_expr = quicksum(fixed_costs[j] * open_facility[j] for j in range(number_of_facilities)) + \
                         quicksum(transportation_costs[j][i] * transport_goods[(i, j)] for i in range(number_of_nodes) for j in range(number_of_facilities))
        objective_expr += quicksum(
            commodities[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i+1}_{j+1}_{k+1}"]
            for (i, j) in edge_list for k in range(len(commodities))
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i+1}_{j+1}"]
            for (i, j) in edge_list
        )
        
        model.setObjective(objective_expr, "minimize")

        # Constraints: Each node's demand must be met
        for i in range(number_of_nodes):
            model.addCons(
                quicksum(transport_goods[(i, j)] for j in range(number_of_facilities)) == node_demands[i],
                f"NodeDemand_{i}"
            )

        # Constraints: Facility capacity must not be exceeded 
        for j in range(number_of_facilities):
            model.addCons(
                quicksum(transport_goods[(i,j)] for i in range(number_of_nodes)) <= facility_capacities[j] * open_facility[j],
                f"FacilityCapacity_{j}"
            )

        # Constraints: Ensure transportation is feasible only if facility is open
        for i in range(number_of_nodes):
            for j in range(number_of_facilities):
                model.addCons(
                    transport_goods[(i, j)] <= M * open_facility[j],
                    f"BigM_TransFeasibility_{i}_{j}"
                )
        
        # Constraints: Node-Facility specific capacity constraints
        for i in range(number_of_nodes):
            for j in range(number_of_facilities):
                model.addCons(
                    transport_goods[(i, j)] <= node_facility_capacities[i][j],
                    f"NodeFacilityCap_{i}_{j}"
                )
        
        # Multicommodity Flow Constraints: Flow conservation for each commodity
        for i in range(self.n_nodes):
            for k in range(len(commodities)):
                delta_i = 1 if commodities[k][0] == i else -1 if commodities[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j+1}_{i+1}_{k+1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i+1}_{k+1}")

        # Multicommodity Arc Capacity Constraints and Complementarity Constraints
        for (i, j) in edge_list:
            arc_expr = quicksum(commodities[k][2] * x_vars[f"x_{i+1}_{j+1}_{k+1}"] for k in range(len(commodities)))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i+1}_{j+1}"], f"arc_{i+1}_{j+1}")

            for k in range(len(commodities)):
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] >= x_vars[f"x_{i+1}_{j+1}_{k+1}"] - (1 - y_vars[f"y_{i+1}_{j+1}"]) * adj_mat[i, j][2], f"comp1_{i+1}_{j+1}_{k+1}")
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] <= x_vars[f"x_{i+1}_{j+1}_{k+1}"], f"comp2_{i+1}_{j+1}_{k+1}")
                model.addCons(z_vars[f"z_{i+1}_{j+1}_{k+1}"] <= y_vars[f"y_{i+1}_{j+1}"] * adj_mat[i, j][2], f"comp3_{i+1}_{j+1}_{k+1}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    parameters = {
        'number_of_facilities': 10,
        'number_of_nodes': 15,
        'min_cost': 50,
        'max_cost': 1000,
        'min_cap': 350,
        'max_cap': 1600,
        'min_demand': 60,
        'max_demand': 300,
        'node_facility_min_cap': 60,
        'node_facility_max_cap': 900,
        'min_n_nodes': 15,
        'max_n_nodes': 20,
        'min_n_commodities': 30,
        'max_n_commodities': 50,
        'c_range': (70, 350),
        'd_range': (200, 2000),
        'ratio': 500,
        'k_max': 60,
        'er_prob': 0.62,
    }

    seed = 42
    facility_location = FacilityLocationTransportation(parameters, seed=seed)
    instance = facility_location.generate_instance()
    solve_status, solve_time = facility_location.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")