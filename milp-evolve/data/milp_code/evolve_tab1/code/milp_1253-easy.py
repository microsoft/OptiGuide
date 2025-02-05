import random
import time
import numpy as np
import networkx as nx
from itertools import combinations
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

class MultipleKnapsackWithMaxSAT:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        weights = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(self.min_range, self.max_range, self.number_of_items)
        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (self.max_range-self.min_range), 1),
                               weights + (self.max_range-self.min_range)]))
        elif self.scheme == 'strongly correlated':
            profits = weights + (self.max_range - self.min_range) / 10
        elif self.scheme == 'subset-sum':
            profits = weights
        else:
            raise NotImplementedError

        capacities = np.zeros(self.number_of_knapsacks, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * weights.sum() // self.number_of_knapsacks,
                                            0.6 * weights.sum() // self.number_of_knapsacks,
                                            self.number_of_knapsacks - 1)
        capacities[-1] = 0.5 * weights.sum() - capacities[:-1].sum()

        distances = np.random.randint(1, 100, (self.number_of_knapsacks, self.number_of_items))
        demand_periods = np.random.randint(1, 10, self.number_of_items)
        res = {'weights': weights, 
               'profits': profits, 
               'capacities': capacities,
               'distances': distances,
               'demand_periods': demand_periods}

        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_maxsat_graph(n)
        clauses = [(f'v{i},v{j}', 1) for i, j in edges] + [(f'-v{i},-v{j}', 1) for i, j in edges]

        # Generate piecewise linear function parameters
        num_pieces = self.num_pieces
        breakpoints = sorted(np.random.uniform(self.min_resource, self.max_resource, num_pieces - 1))
        slopes = np.random.uniform(self.min_slope, self.max_slope, num_pieces)
        intercepts = np.random.uniform(self.min_intercept, self.max_intercept, num_pieces)

        res.update({
            'clauses': clauses,
            'breakpoints': breakpoints,
            'slopes': slopes,
            'intercepts': intercepts
        })

        # Generate additional instance data for Patient Coverage
        pairs = self.generate_patient_doctor_pairs()
        patient_capacities = np.random.randint(5, 20, size=self.n_doctors)
        res.update({
            'patient_doctor_pairs': pairs,
            'doctor_capacities': patient_capacities,
            'n_doctors': self.n_doctors,
            'n_patients': self.n_patients,
        })

        # Generate additional instance data for new constraints
        fuel_efficiency = np.random.uniform(0.8, 1.2, self.number_of_knapsacks)
        zone_restrictions = np.random.choice([0, 1], (self.number_of_knapsacks, self.number_of_zones))
        res.update({
            'fuel_efficiency': fuel_efficiency,
            'zone_restrictions': zone_restrictions,
        })

        return res

    def generate_bipartite_graph(self, n1, n2, p):
        return bipartite.random_graph(n1, n2, p, seed=self.seed)

    def generate_maxsat_graph(self, n):
        divider = np.random.randint(1, 6)
        G = self.generate_bipartite_graph(n // divider, n - n // divider, self.er_prob)

        n_edges = len(G.edges)
        edges = list(G.edges)

        added_edges = 0
        while added_edges < n_edges * self.edge_addition_prob:
            i, j = np.random.randint(0, n), np.random.randint(0, n)
            if (i, j) not in edges and (j, i) not in edges:
                added_edges += 1
                edges.append((i, j))
        return edges

    def generate_patient_doctor_pairs(self):
        pairs = set()
        for patient in range(self.n_patients):
            num_pairs = np.random.randint(1, self.n_doctors // 2)
            doctors = np.random.choice(self.n_doctors, num_pairs, replace=False)
            for doctor in doctors:
                pairs.add((patient, doctor))
        return pairs

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        distances = instance['distances']
        demand_periods = instance['demand_periods']
        clauses = instance['clauses']
        breakpoints = instance['breakpoints']
        slopes = instance['slopes']
        intercepts = instance['intercepts']
        patient_doctor_pairs = instance['patient_doctor_pairs']
        doctor_capacities = instance['doctor_capacities']
        n_doctors = instance['n_doctors']
        n_patients = instance['n_patients']
        fuel_efficiency = instance['fuel_efficiency']
        zone_restrictions = instance['zone_restrictions']

        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)

        model = Model("MultipleKnapsackWithMaxSAT")
        var_names = {}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Decision variables: t[j] = total time for deliveries by vehicle j
        time_vars = {}
        for j in range(number_of_knapsacks):
            time_vars[j] = model.addVar(vtype="C", name=f"t_{j}")

        # Decision variables: y[j] = 1 if vehicle j is used
        vehicle_vars = {}
        for j in range(number_of_knapsacks):
            vehicle_vars[j] = model.addVar(vtype="B", name=f"y_{j}")

        # Decision variables: f[j] = total fuel consumed by knapsack j
        fuel_vars = {}
        for j in range(number_of_knapsacks):
            fuel_vars[j] = model.addVar(vtype="C", name=f"f_{j}")

        # Decision variables: m[j] = 1 if plane j is in maintenance
        maintenance_vars = {}
        for j in range(number_of_knapsacks):
            maintenance_vars[j] = model.addVar(vtype="B", name=f"m_{j}")

        # Decision variables: z[j][k] = 1 if knapsack j is allowed in zone k
        zoning_vars = {}
        for j in range(number_of_knapsacks):
            for k in range(self.number_of_zones):
                zoning_vars[(j, k)] = model.addVar(vtype="B", name=f"z_{j}_{k}")

        # Objective: Maximize total profit - minimize fuel and maintenance costs
        objective_expr = (
            quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)) -
            quicksum(self.fuel_costs * distances[j][i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)) -
            quicksum(self.vehicle_usage_costs * vehicle_vars[j] for j in range(number_of_knapsacks)) -
            quicksum(self.maintenance_costs[j] * maintenance_vars[j] for j in range(number_of_knapsacks))
        )

        # Constraints: Each item can be in at most one knapsack
        for i in range(number_of_items):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_knapsacks)) <= 1,
                f"ItemAssignment_{i}"
            )

        # Constraints: Total weight in each knapsack must not exceed its capacity
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(weights[i] * var_names[(i, j)] for i in range(number_of_items)) <= capacities[j],
                f"KnapsackCapacity_{j}"
            )

        # Constraints: Total delivery time for each vehicle must not exceed the time limit
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(demand_periods[i] * var_names[(i, j)] for i in range(number_of_items)) <= self.time_limit,
                f"DeliveryTime_{j}"
            )

        # Constraints: Ensure if a vehicle is used, it must be assigned exactly one driver
        for j in range(number_of_knapsacks):
            model.addCons(
                quicksum(var_names[(i, j)] for i in range(number_of_items)) >= vehicle_vars[j],
                f"VehicleUsed_{j}"
            )

        # Fuel consumption constraints for each knapsack
        for j in range(number_of_knapsacks):
            model.addCons(
                fuel_vars[j] == quicksum(distances[j][i] * var_names[(i, j)] * fuel_efficiency[j] for i in range(number_of_items)),
                name=f"FuelConsumption_{j}"
            )

        model.addCons(
            quicksum(fuel_vars[j] for j in range(number_of_knapsacks)) <= self.fuel_limit,
            name="TotalFuelLimit"
        )

        # Maintenance constraints
        for j in range(number_of_knapsacks):
            for i in range(number_of_items):
                model.addCons(
                    var_names[(i, j)] + maintenance_vars[j] <= 1,
                    name=f"MaintenanceRestraint_{j}_{i}"
                )

        # Zoning constraints
        for j in range(number_of_knapsacks):
            for i in range(number_of_items):
                for k in range(self.number_of_zones):
                    model.addCons(
                        var_names[(i, j)] <= zoning_vars[(j, k)] * zone_restrictions[j][k],
                        name=f"ZoneRestriction_{j}_{i}_{k}"
                    )

        # Create variables for each literal and clause
        literal_vars = {}  
        clause_vars = {}

        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in literal_vars:
                    literal_vars[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            clause_vars[f"cl_{idx}"] = clause_var

        # Create resource variable
        resource = model.addVar(vtype="C", lb=self.min_resource, ub=self.max_resource, name="resource")

        # Piecewise linear cost function
        cost = model.addVar(vtype="C", name="cost")
        for i in range(len(breakpoints) - 1):
            model.addCons(
                cost >= slopes[i] * resource + intercepts[i],
                name=f"cost_piece_{i}")
        
        model.addCons(
            cost >= slopes[-1] * resource + intercepts[-1],
            name="cost_last_piece")

        # Update objective function to include clauses and cost
        objective_expr += quicksum(
            clause_vars[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        ) - cost

        # Add constraints for each clause
        for idx, (clause, weight) in enumerate(clauses):
            vars_in_clause = clause.split(',')
            clause_var = clause_vars[f"cl_{idx}"]
            
            positive_part = quicksum(literal_vars[var] for var in vars_in_clause if not var.startswith('-'))
            negative_part = quicksum(1 - literal_vars[var[1:]] for var in vars_in_clause if var.startswith('-'))
            total_satisfied = positive_part + negative_part
            
            if weight < np.inf:
                model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")
            else:
                model.addCons(total_satisfied >= 1, name=f"clause_{idx}")

        ## New constraints and variables for Patient Coverage MILP
        ## Doctor variables: d[j] = number of patients covered by doctor j
        doctor_vars = {}
        for doctor in range(n_doctors):
            doctor_vars[doctor] = model.addVar(vtype="I", lb=0, ub=doctor_capacities[doctor], name=f"d_{doctor}")

        ## Patient coverage: p[i] = 1 if patient i is covered by any doctor
        patient_covered = {}
        for patient in range(n_patients):
            patient_covered[patient] = model.addVar(vtype="B", name=f"p_{patient}")

        ## Each patient's coverage must not exceed doctor's capacity
        for patient, doctor in patient_doctor_pairs:
            model.addCons(patient_covered[patient] <= doctor_vars[doctor], name=f"coverage_{patient}_{doctor}")

        ## Ensure doctor's capacity limit is not exceeded
        for doctor in range(n_doctors):
            model.addCons(quicksum(patient_covered[patient] for patient, doc in patient_doctor_pairs if doc == doctor) <= doctor_vars[doctor], name=f"capacity_{doctor}")

        ## Update objective function to include patient coverage
        objective_expr += quicksum(patient_covered[patient] for patient in range(n_patients))

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 52,
        'number_of_knapsacks': 10,
        'number_of_zones': 5,
        'min_range': 1,
        'max_range': 225,
        'scheme': 'weakly correlated',
        'fuel_limit': 2000,
        'fuel_costs': 35,
        'vehicle_usage_costs': 75,
        'maintenance_costs': [random.randint(50, 100) for _ in range(10)],
        'time_limit': 3000,
        'min_n': 28,
        'max_n': 93,
        'er_prob': 0.45,
        'edge_addition_prob': 0.1,
        'num_pieces': 15,
        'min_resource': 0,
        'max_resource': 125,
        'min_slope': 0,
        'max_slope': 500,
        'min_intercept': 0,
        'max_intercept': 65,
        'n_doctors': 25,
        'n_patients': 750,
    }

    knapsack_with_maxsat = MultipleKnapsackWithMaxSAT(parameters, seed=seed)
    instance = knapsack_with_maxsat.generate_instance()
    solve_status, solve_time = knapsack_with_maxsat.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")