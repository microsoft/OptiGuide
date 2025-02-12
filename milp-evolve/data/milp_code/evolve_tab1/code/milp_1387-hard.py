import random
import time
import scipy
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ChargingStationNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        nnzrs = int(self.n_rows * self.n_cols * self.density)

        indices = np.random.choice(self.n_cols, size=nnzrs)
        indices[:2 * self.n_cols] = np.repeat(np.arange(self.n_cols), 2)
        _, col_nrows = np.unique(indices, return_counts=True)

        indices[:self.n_rows] = np.random.permutation(self.n_rows)
        i = 0
        indptr = [0]
        for n in col_nrows:
            if i >= self.n_rows:
                indices[i:i+n] = np.random.choice(self.n_rows, size=n, replace=False)
            elif i + n > self.n_rows:
                remaining_rows = np.setdiff1d(np.arange(self.n_rows), indices[i:self.n_rows], assume_unique=True)
                indices[self.n_rows:i+n] = np.random.choice(remaining_rows, size=i+n-self.n_rows, replace=False)
            i += n
            indptr.append(i)

        c = np.random.randint(self.max_coef, size=self.n_cols) + 1
        habitat_dists = np.random.randint(self.max_habitat_dist+1, size=(self.n_cols, self.n_habitats))

        A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(self.n_rows, self.n_cols)).tocsr()
        indices_csr = A.indices
        indptr_csr = A.indptr

        # New data for traffic patterns, energy demands, renewable supply
        traffic_matrix = np.random.rand(self.n_stations, self.n_stations)
        energy_demand = np.random.randint(5, 15, size=self.n_vehicle_models)
        renewable_supply = np.random.randint(200, 500, size=self.n_stations)
        charging_time = np.random.randint(30, 60, size=self.n_vehicle_models)

        res = {
            'c': c, 
            'indptr_csr': indptr_csr, 
            'indices_csr': indices_csr,
            'habitat_dists': habitat_dists,
            'traffic_matrix': traffic_matrix,
            'energy_demand': energy_demand,
            'renewable_supply': renewable_supply,
            'charging_time': charging_time
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        c = instance['c']
        indptr_csr = instance['indptr_csr']
        indices_csr = instance['indices_csr']
        habitat_dists = instance['habitat_dists']
        traffic_matrix = instance['traffic_matrix']
        energy_demand = instance['energy_demand']
        renewable_supply = instance['renewable_supply']
        charging_time = instance['charging_time']

        model = Model("ChargingStationNetwork")
        var_names = {}
        env_vars = []
        traffic_vars = []

        # Create variables and set objective
        for j in range(self.n_cols):
            var_names[j] = model.addVar(vtype="B", name=f"x_{j}", obj=c[j])
            env_vars.append(model.addVar(vtype="B", name=f"y_{j}"))
        
        for s in range(self.n_stations):
            traffic_vars.append(model.addVar(vtype="C", lb=0, name=f"traffic_{s}"))

        # Add constraints to ensure each row is covered
        for row in range(self.n_rows):
            cols = indices_csr[indptr_csr[row]:indptr_csr[row + 1]]
            model.addCons(quicksum(var_names[j] for j in cols) >= 1, f"c_{row}")
        
        # Environmental constraints ensuring no tower is in a prohibited distance
        for j in range(self.n_cols):
            for habitat in range(self.n_habitats):
                if habitat_dists[j][habitat] < self.eco_dist:
                    model.addCons(env_vars[j] == 0, f"env_{j}_{habitat}")
        
        # Traffic impact constraints on station utilization
        for station in range(self.n_stations):
            model.addCons(quicksum(traffic_matrix[station][k] * traffic_vars[k] for k in range(self.n_stations)) <= self.max_traffic, f"TrafficImpact_{station}")

        # Energy balance constraints
        for station in range(self.n_stations):
            energy_consumed = quicksum(var_names[vehicle] * energy_demand[vehicle] for vehicle in range(self.n_vehicle_models))
            model.addCons(energy_consumed <= renewable_supply[station], f"EnergyBalance_{station}")

        # Charging time constraints to ensure different vehicle models receive the appropriate time
        for vehicle in range(self.n_vehicle_models):
            model.addCons(var_names[vehicle] * charging_time[vehicle] >= 0, f"ChargingTime_{vehicle}")

        # Budget constraint
        total_cost = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        model.addCons(total_cost <= self.budget, "Budget")
        
        # Set objective: Minimize total cost
        objective_expr = quicksum(var_names[j] * c[j] for j in range(self.n_cols))
        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_rows': 2250,
        'n_cols': 750,
        'density': 0.17,
        'max_coef': 350,
        'max_habitat_dist': 111,
        'eco_dist': 200,
        'budget': 5000,
        'n_habitats': 50,
        'n_stations': 750,
        'n_vehicle_models': 20,
        'max_traffic': 7.0,
    }

    charging_station_problem = ChargingStationNetwork(parameters, seed=seed)
    instance = charging_station_problem.generate_instance()
    solve_status, solve_time = charging_station_problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")