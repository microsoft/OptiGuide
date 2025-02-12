import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class EmergencyResponseMILP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_erdos_graph(self):
        G = nx.erdos_renyi_graph(n=self.n_regions, p=self.er_prob, seed=self.seed, directed=True)
        adj_mat = np.zeros((self.n_regions, self.n_regions), dtype=object)
        edge_list = []
        incommings = {j: [] for j in range(self.n_regions)}
        outcommings = {i: [] for i in range(self.n_regions)}

        for i, j in G.edges():
            c_ij = np.random.uniform(*self.transport_cost_range)
            f_ij = np.random.uniform(self.transport_cost_range[0] * self.cost_ratio, self.transport_cost_range[1] * self.cost_ratio)
            u_ij = np.random.uniform(1, self.max_routes + 1) * np.random.uniform(*self.demand_range)
            adj_mat[i, j] = (c_ij, f_ij, u_ij)
            edge_list.append((i, j))
            outcommings[i].append(j)
            incommings[j].append(i)

        return G, adj_mat, edge_list, incommings, outcommings

    def generate_demands(self, G):
        demands = []
        for k in range(self.n_demands):
            while True:
                o_k = np.random.randint(0, self.n_regions)
                d_k = np.random.randint(0, self.n_regions)
                if nx.has_path(G, o_k, d_k) and o_k != d_k:
                    break
            demand_k = int(np.random.uniform(*self.demand_range))
            demands.append((o_k, d_k, demand_k))
        return demands

    def generate_healthcare_data(self):
        healthcare_costs = np.random.randint(1, self.healthcare_max_cost + 1, size=self.n_regions)
        healthcare_pairs = [(i, j) for i in range(self.n_regions) for j in range(self.n_regions) if i != j]
        chosen_pairs = np.random.choice(len(healthcare_pairs), size=self.n_healthcare_pairs, replace=False)
        healthcare_set = [healthcare_pairs[i] for i in chosen_pairs]
        return healthcare_costs, healthcare_set

    def generate_shelter_data(self):
        shelter_costs = np.random.randint(1, self.shelter_max_cost + 1, size=self.n_regions)
        shelter_caps = np.random.randint(1, self.shelter_max_cap + 1, size=self.n_regions)
        return shelter_costs, shelter_caps

    def generate_labor_shifts(self):
        shifts = np.random.randint(1, self.max_shifts + 1)
        shift_capacity = np.random.randint(1, self.max_shift_capacity + 1, size=shifts)
        return shifts, shift_capacity

    def generate_instance(self):
        self.n_regions = np.random.randint(self.min_n_regions, self.max_n_regions+1)
        self.n_demands = np.random.randint(self.min_n_demands, self.max_n_demands + 1)
        G, adj_mat, edge_list, incommings, outcommings = self.generate_erdos_graph()
        demands = self.generate_demands(G)
        healthcare_costs, healthcare_set = self.generate_healthcare_data()
        shelter_costs, shelter_caps = self.generate_shelter_data()
        shifts, shift_capacity = self.generate_labor_shifts()
        capacity_thresh = np.random.randint(self.min_capacity_thresh, self.max_capacity_thresh)
        set_covering_needs = np.random.randint(1, self.set_covering_max_need + 1, size=self.n_regions)
        
        res = {
            'demands': demands, 
            'adj_mat': adj_mat, 
            'edge_list': edge_list, 
            'incommings': incommings, 
            'outcommings': outcommings, 
            'healthcare_costs': healthcare_costs, 
            'healthcare_set': healthcare_set,
            'shelter_costs': shelter_costs,
            'shelter_caps': shelter_caps,
            'shifts': shifts,
            'shift_capacity': shift_capacity,
            'capacity_thresh': capacity_thresh,
            'set_covering_needs': set_covering_needs
        }
        
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        demands = instance['demands']
        adj_mat = instance['adj_mat']
        edge_list = instance['edge_list']
        incommings = instance['incommings']
        outcommings = instance['outcommings']
        healthcare_costs = instance['healthcare_costs']
        healthcare_set = instance['healthcare_set']
        shelter_costs = instance['shelter_costs']
        shelter_caps = instance['shelter_caps']
        shifts = instance['shifts']
        shift_capacity = instance['shift_capacity']
        capacity_thresh = instance['capacity_thresh']
        set_covering_needs = instance['set_covering_needs']
        
        model = Model("EmergencyResponseMILP")
        x_vars = {f"x_{i + 1}_{j + 1}_{k + 1}": model.addVar(vtype="C", name=f"x_{i + 1}_{j + 1}_{k + 1}") for (i, j) in edge_list for k in range(self.n_demands)}
        y_vars = {f"y_{i + 1}_{j + 1}": model.addVar(vtype="B", name=f"y_{i + 1}_{j + 1}") for (i, j) in edge_list}
        z_vars = {f"z_{i + 1}_{j + 1}": model.addVar(vtype="B", name=f"z_{i + 1}_{j + 1}") for (i, j) in healthcare_set}
        w_vars = {f"w_{i + 1}": model.addVar(vtype="B", name=f"w_{i + 1}") for i in range(self.n_regions)}

        # New variables
        shift_vars = {f"shift_{m + 1}_{s + 1}": model.addVar(vtype="B", name=f"shift_{m + 1}_{s + 1}") for m in range(self.n_regions) for s in range(shifts)}
        usage_vars = {f"usage_{m + 1}_{t + 1}": model.addVar(vtype="C", name=f"usage_{m + 1}_{t + 1}") for m in range(self.n_regions) for t in range(self.n_regions)}
        capacity_vars = {f"capacity_{m + 1}": model.addVar(vtype="B", name=f"capacity_{m + 1}") for m in range(self.n_regions)}
        aid_vars = {f"aid_{k + 1}": model.addVar(vtype="B", name=f"aid_{k + 1}") for k in range(self.n_demands)}
        healthcare_allocation_vars = {f"health_alloc_{r + 1}_{h + 1}": model.addVar(vtype="B", name=f"health_alloc_{r + 1}_{h + 1}") for r in range(self.n_regions) for h in range(self.n_healthcare_units)}
        worker_shift_vars = {f"worker_shift_{w + 1}_{s + 1}": model.addVar(vtype="B", name=f"worker_shift_{w + 1}_{s + 1}") for w in range(self.n_workers) for s in range(shifts)}

        objective_expr = quicksum(
            demands[k][2] * adj_mat[i, j][0] * x_vars[f"x_{i + 1}_{j + 1}_{k + 1}"]
            for (i, j) in edge_list for k in range(self.n_demands)
        )
        objective_expr += quicksum(
            adj_mat[i, j][1] * y_vars[f"y_{i + 1}_{j + 1}"]
            for (i, j) in edge_list
        )
        objective_expr += quicksum(
            healthcare_costs[i] * z_vars[f"z_{i + 1}_{j + 1}"]
            for (i, j) in healthcare_set
        )
        objective_expr += quicksum(
            shelter_costs[i] * w_vars[f"w_{i + 1}"]
            for i in range(self.n_regions)
        )
        objective_expr += quicksum(
            shift_capacity[s] * shift_vars[f"shift_{m + 1}_{s + 1}"]
            for m in range(self.n_regions) for s in range(shifts)
        )

        for i in range(self.n_regions):
            for k in range(self.n_demands):
                delta_i = 1 if demands[k][0] == i else -1 if demands[k][1] == i else 0
                flow_expr = quicksum(x_vars[f"x_{i + 1}_{j + 1}_{k + 1}"] for j in outcommings[i]) - quicksum(x_vars[f"x_{j + 1}_{i + 1}_{k + 1}"] for j in incommings[i])
                model.addCons(flow_expr == delta_i, f"flow_{i + 1}_{k + 1}")

        for (i, j) in edge_list:
            arc_expr = quicksum(demands[k][2] * x_vars[f"x_{i + 1}_{j + 1}_{k + 1}"] for k in range(self.n_demands))
            model.addCons(arc_expr <= adj_mat[i, j][2] * y_vars[f"y_{i + 1}_{j + 1}"], f"arc_{i + 1}_{j + 1}")

        for i in range(self.n_regions):
            healthcare_expr = quicksum(z_vars[f"z_{j + 1}_{i + 1}"] for j in range(self.n_regions) if (j, i) in healthcare_set)
            model.addCons(healthcare_expr >= 1, f"healthcare_{i + 1}")

        for i in range(self.n_regions):
            shelter_capacity_expr = quicksum(demands[k][2] * x_vars[f"x_{i + 1}_{j + 1}_{k + 1}"] for j in outcommings[i] for k in range(self.n_demands)) 
            model.addCons(shelter_capacity_expr <= shelter_caps[i] + w_vars[f'w_{i + 1}'] * sum(demands[k][2] for k in range(self.n_demands)), f"shelter_{i + 1}")

        # New constraints for shift scheduling
        for m in range(self.n_regions):
            model.addCons(quicksum(shift_vars[f"shift_{m + 1}_{s + 1}"] for s in range(shifts)) <= 1, f"shift_allocation_{m + 1}")

        for s in range(shifts):
            shift_labor_expr = quicksum(shift_vars[f"shift_{m + 1}_{s + 1}"] for m in range(self.n_regions))
            model.addCons(shift_labor_expr <= shift_capacity[s], f"shift_capacity_{s + 1}")

        # New constraints for shelter capacity
        for m in range(self.n_regions):
            cum_usage_expr = quicksum(usage_vars[f"usage_{m + 1}_{t + 1}"] for t in range(self.n_regions))
            model.addCons(cum_usage_expr <= capacity_thresh * (1 - capacity_vars[f"capacity_{m + 1}"]), f"shelter_capacity_thresh_{m + 1}")

        # Set covering constraints for healthcare allocation
        for r in range(self.n_regions):
            healthcare_cover_expr = quicksum(healthcare_allocation_vars[f"health_alloc_{r + 1}_{h + 1}"] for h in range(self.n_healthcare_units))
            model.addCons(healthcare_cover_expr >= set_covering_needs[r], f"healthcare_set_cover_{r + 1}")

        # Set covering constraints for worker allocation to shifts
        for s in range(shifts):
            worker_shift_cover_expr = quicksum(worker_shift_vars[f"worker_shift_{w + 1}_{s + 1}"] for w in range(self.n_workers))
            model.addCons(worker_shift_cover_expr >= 1, f"worker_shift_cover_{s + 1}")

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time
    

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_regions': 20,
        'max_n_regions': 30,
        'min_n_demands': 30,
        'max_n_demands': 45, 
        'transport_cost_range': (11, 50),
        'demand_range': (10, 100),
        'cost_ratio': 100,
        'max_routes': 10,
        'er_prob': 0.3,
        'n_healthcare_pairs': 50,
        'healthcare_max_cost': 20,
        'shelter_max_cost': 30,
        'shelter_max_cap': 200,
        'min_capacity_thresh': 30,
        'max_capacity_thresh': 50,
        'max_shifts': 3,
        'max_shift_capacity': 5,
        'n_healthcare_units': 10,
        'n_workers': 100,
        'set_covering_max_need': 5
    }

    response_model = EmergencyResponseMILP(parameters, seed=seed)
    instance = response_model.generate_instance()
    solve_status, solve_time = response_model.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")