import random
import time
import numpy as np
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

class EVFleetChargingOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_bipartite_graph(self, n1, n2, p):
        return bipartite.random_graph(n1, n2, p, seed=self.seed)

    def generate_instance(self):
        nnzrs = int(self.n_vehicles * self.n_stations * self.density)
        
        # Vehicle power requirements and station power capacities
        vehicle_power_req = np.random.randint(self.min_power_req, self.max_power_req, size=self.n_vehicles)
        station_power_cap = np.random.randint(self.min_power_cap, self.max_power_cap, size=self.n_stations)
        
        # Charging efficiency and cost
        charging_efficiency = np.random.uniform(self.min_efficiency, self.max_efficiency, size=(self.n_stations, self.n_vehicles))
        station_activation_cost = np.random.randint(self.activation_cost_low, self.activation_cost_high, size=self.n_stations)
        
        # Max-SAT logical requirements
        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_bipartite_graph(n // self.divider, n - (n // self.divider), self.er_prob)
        
        clauses = [(f'v{i}_s{j}', 1) for i, j in edges.edges] + [(f'-v{i}_s{j}', 1) for i, j in edges.edges]

        res = {
            'vehicle_power_req': vehicle_power_req,
            'station_power_cap': station_power_cap,
            'charging_efficiency': charging_efficiency,
            'station_activation_cost': station_activation_cost,
            'clauses': clauses,
        }

        # Environmental impact and transportation modes
        environmental_impact = np.random.randint(self.min_env_impact, self.max_env_impact, size=(self.n_stations, self.n_vehicles))
        transport_modes = np.random.randint(self.min_transport_modes, self.max_transport_modes, size=self.n_transport_modes)
        
        res['environmental_impact'] = environmental_impact
        res['transport_modes'] = transport_modes

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        vehicle_power_req = instance['vehicle_power_req']
        station_power_cap = instance['station_power_cap']
        charging_efficiency = instance['charging_efficiency']
        station_activation_cost = instance['station_activation_cost']
        clauses = instance['clauses']
        environmental_impact = instance['environmental_impact']
        transport_modes = instance['transport_modes']

        model = Model("EVFleetChargingOptimization")
        station_vars = {}
        allocation_vars = {}
        charge_level_vars = {}
        satisfiability_vars = {}
        clause_satisfaction_lvl = {}
        environmental_impact_vars = {}

        # Create variables and set objectives
        for j in range(self.n_stations):
            station_vars[j] = model.addVar(vtype="B", name=f"Station_{j}", obj=station_activation_cost[j])

        for v in range(self.n_vehicles):
            for s in range(self.n_stations):
                allocation_vars[(v, s)] = model.addVar(vtype="B", name=f"Vehicle_{v}_Station_{s}", obj=charging_efficiency[s][v] * vehicle_power_req[v])

        for s in range(self.n_stations):
            charge_level_vars[s] = model.addVar(vtype="C", name=f"Charge_Level_{s}")

        for t in range(self.n_transport_modes):
            environmental_impact_vars[t] = model.addVar(vtype="C", name=f"Environmental_Impact_Mode_{t}")

        # Ensure each vehicle is assigned to one station
        for v in range(self.n_vehicles):
            model.addCons(quicksum(allocation_vars[(v, s)] for s in range(self.n_stations)) == 1, f"Vehicle_{v}_Assignment")

        # Power capacity constraints for stations
        for s in range(self.n_stations):
            model.addCons(quicksum(allocation_vars[(v, s)] * vehicle_power_req[v] for v in range(self.n_vehicles)) <= station_power_cap[s], f"Station_{s}_Capacity")
            model.addCons(charge_level_vars[s] == quicksum(allocation_vars[(v, s)] * vehicle_power_req[v] for v in range(self.n_vehicles)), f"Charge_Level_{s}")
            model.addCons(charge_level_vars[s] <= station_power_cap[s] * station_vars[s], f"Station_{s}_Active_Capacity")
        
        # Environmental impact constraints
        for t in range(self.n_transport_modes):
            model.addCons(quicksum(environmental_impact[s, v] * allocation_vars[(v, s)] for s in range(self.n_stations) for v in range(self.n_vehicles)) <= environmental_impact_vars[t], f"Environmental_Impact_Mode_{t}")

        # Satisfiability constraints
        for idx, (clause, weight) in enumerate(clauses):
            var_names = clause.split(',')
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            satisfiability_vars[f"cl_{idx}"] = clause_var
            
            positive_part = quicksum(allocation_vars[int(var.split('_')[0][1:]), int(var.split('_')[1][1:])] for var in var_names if not var.startswith('-'))
            negative_part = quicksum(1 - allocation_vars[int(var.split('_')[0][2:]), int(var.split('_')[1][1:])] for var in var_names if var.startswith('-'))
            
            total_satisfied = positive_part + negative_part
            model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")

        L, U = self.semi_cont_l, self.semi_cont_u
        for idx in range(len(clauses)):
            clause_satisfaction_lvl[f"cl_s_lvl_{idx}"] = model.addVar(lb=0, ub=U, vtype="CONTINUOUS", name=f"cl_s_lvl_{idx}") 
            model.addCons(clause_satisfaction_lvl[f"cl_s_lvl_{idx}"] >= L * satisfiability_vars[f"cl_{idx}"], name=f"L_bound_s_lvl_{idx}")
            model.addCons(clause_satisfaction_lvl[f"cl_s_lvl_{idx}"] <= U * satisfiability_vars[f"cl_{idx}"], name=f"U_bound_s_lvl_{idx}")

        # Objective: Minimize total cost including energy usage, station activation costs, and environmental impact
        objective_expr = quicksum(station_activation_cost[j] * station_vars[j] for j in range(self.n_stations)) + \
                         quicksum(charging_efficiency[s][v] * allocation_vars[(v, s)] * vehicle_power_req[v] for s in range(self.n_stations) for v in range(self.n_vehicles)) + \
                         quicksum(satisfiability_vars[f"cl_{idx}"] for idx in range(len(clauses)))  # Maximizing satisfaction

        objective_expr += quicksum(environmental_impact_vars[t] for t in range(self.n_transport_modes))

        model.setObjective(objective_expr, "minimize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_vehicles': 100,
        'n_stations': 50,
        'density': 0.79,
        'min_power_req': 45,
        'max_power_req': 200,
        'min_power_cap': 700,
        'max_power_cap': 1000,
        'min_efficiency': 0.8,
        'max_efficiency': 2.0,
        'activation_cost_low': 100,
        'activation_cost_high': 1000,
        'min_n': 37,
        'max_n': 250,
        'er_prob': 0.17,
        'edge_addition_prob': 0.66,
        'divider': 3,
        'semi_cont_l': 0,
        'semi_cont_u': 5,
        'min_env_impact': 5,
        'max_env_impact': 20,
        'min_transport_modes': 2,
        'max_transport_modes': 5,
        'n_transport_modes': 3,
    }

    problem = EVFleetChargingOptimization(parameters, seed=seed)
    instance = problem.generate_instance()
    solve_status, solve_time = problem.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")