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
        
        # Generate weather impact factors on capacities
        weather_impacts = np.random.uniform(0.8, 1.0, self.n_facilities)

        # Costs and benefits related to sensor installations
        sensor_costs = np.random.randint(self.min_sensor_cost, self.max_sensor_cost + 1, self.n_nodes)
        water_benefits = np.random.uniform(self.min_water_benefit, self.max_water_benefit, self.n_nodes)

        # Additional data for zero-emission and regional compliance
        z_compliance_costs = np.random.uniform(self.min_z_cost, self.max_z_cost, self.n_facilities)
        r_compliance_costs = np.random.uniform(self.min_r_cost, self.max_r_cost, self.n_facilities)
        zero_emission_benefits = np.random.uniform(self.min_z_benefit, self.max_z_benefit, self.n_facilities)
        regional_penalties = np.random.uniform(self.min_r_penalty, self.max_r_penalty, self.n_facilities)
        
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
            'weather_impacts': weather_impacts,
            'sensor_costs': sensor_costs,
            'water_benefits': water_benefits,
            'z_compliance_costs': z_compliance_costs,
            'r_compliance_costs': r_compliance_costs,
            'zero_emission_benefits': zero_emission_benefits,
            'regional_penalties': regional_penalties
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
        weather_impacts = instance['weather_impacts']
        sensor_costs = instance['sensor_costs']
        water_benefits = instance['water_benefits']
        z_compliance_costs = instance['z_compliance_costs']
        r_compliance_costs = instance['r_compliance_costs']
        zero_emission_benefits = instance['zero_emission_benefits']
        regional_penalties = instance['regional_penalties']

        model = Model("FCMCNF")
        x_vars = {f"x_{i+1}_{j+1}_{k+1}": model.addVar(vtype="C", name=f"x_{i+1}_{j+1}_{k+1}") for (i, j) in edge_list for k in range(self.n_commodities)}
        y_vars = {f"y_{i+1}_{j+1}": model.addVar(vtype="B", name=f"y_{i+1}_{j+1}") for (i, j) in edge_list}
        
        open_vars = {i: model.addVar(vtype="B", name=f"Facility_{i+1}") for i in range(self.n_nodes)}
        assign_vars = {(i, j): model.addVar(vtype="C", name=f"Assign_{i+1}_{j+1}") for i in range(self.n_nodes) for j in range(self.n_nodes)}

        # New variables for sensor installation, water usage, and compliance
        w_vars = {i: model.addVar(vtype="B", name=f"Sensor_{i+1}") for i in range(self.n_nodes)}
        z_vars = {i: model.addVar(vtype="C", name=f"Water_{i+1}") for i in range(self.n_nodes)}
        z_compliance_vars = {i: model.addVar(vtype="B", name=f"Z_Compliance_{i+1}") for i in range(self.n_nodes)}
        r_compliance_vars = {i: model.addVar(vtype="B", name=f"R_Compliance_{i+1}") for i in range(self.n_nodes)}

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

        # Adding sensor costs to the objective
        objective_expr += quicksum(
            sensor_costs[i] * w_vars[i]
            for i in range(self.n_nodes)
        )

        # Adding water benefits to the objective
        objective_expr -= quicksum(
            water_benefits[i] * z_vars[i]
            for i in range(self.n_nodes)
        )

        # Adding zero-emission compliance costs to the objective
        objective_expr += quicksum(
            z_compliance_costs[i] * z_compliance_vars[i]
            for i in range(self.n_nodes)
        )

        # Adding regional compliance costs and penalties to the objective
        objective_expr += quicksum(
            r_compliance_costs[i] * r_compliance_vars[i]
            for i in range(self.n_nodes)
        )
        objective_expr += quicksum(
            regional_penalties[i] * (1 - r_compliance_vars[i])
            for i in range(self.n_nodes)
        )

        # Adding zero-emission benefits to the objective
        objective_expr -= quicksum(
            zero_emission_benefits[i] * z_compliance_vars[i]
            for i in range(self.n_nodes)
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
            # Capacity considering weather impacts
            model.addCons(quicksum(assign_vars[(i, j)] for j in range(self.n_nodes)) <= capacities[i] * weather_impacts[i], f"Facility_{i+1}_Capacity")
            model.addCons(quicksum(assign_vars[(i, j)] for j in range(self.n_nodes)) <= open_vars[i] * np.sum(demands), f"Facility_{i+1}_Open")

        for j in range(self.n_nodes):
            model.addCons(quicksum(assign_vars[(i, j)] for i in range(self.n_nodes)) == demands[j], f"Demand_{j+1}")

        # Water usage constraints
        for i in range(self.n_nodes):
            model.addCons(z_vars[i] <= 10 * w_vars[i], f"WaterUsage_{i+1}")  # if sensor installed, can have up to 10 units of water
            model.addCons(z_vars[i] <= 5, f"WaterUsageMax_{i+1}")           # 5 units of water if no sensor

        # Zero-emission and regional compliance constraints
        for i in range(self.n_nodes):
            model.addCons(quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i] for k in range(self.n_commodities)) >= z_compliance_vars[i], f"Z_Emission_Compliance_{i+1}")
            model.addCons(quicksum(x_vars[f"x_{i+1}_{j+1}_{k+1}"] for j in outcommings[i] for k in range(self.n_commodities)) >= r_compliance_vars[i], f"R_Emission_Compliance_{i+1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 15,
        'max_n_nodes': 15,
        'min_n_commodities': 150,
        'max_n_commodities': 675,
        'c_range': (110, 500),
        'd_range': (5, 50),
        'ratio': 50,
        'k_max': 70,
        'er_prob': 0.45,
        'min_facility_cost': 15,
        'max_facility_cost': 750,
        'min_region_benefit': 10,
        'max_region_benefit': 600,
        'min_facility_cap': 37,
        'max_facility_cap': 900,
        'min_sensor_cost': 5,
        'max_sensor_cost': 15,
        'min_water_benefit': 5,
        'max_water_benefit': 150,
        'min_z_cost': 1000,
        'max_z_cost': 5000,
        'min_r_cost': 250,
        'max_r_cost': 1500,
        'min_z_benefit': 30,
        'max_z_benefit': 50,
        'min_r_penalty': 37,
        'max_r_penalty': 1400,
    }

    fcmcnf = FCMCNF(parameters, seed=seed)
    instance = fcmcnf.generate_instance()
    solve_status, solve_time = fcmcnf.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")