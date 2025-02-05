import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum
from networkx.algorithms import bipartite

class ComplexSupplyChainAndEVChargingOptimization:
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

    def generate_instances(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        edges = self.generate_maxsat_graph(n)
        clauses = [(f'v{i},v{j}', 1) for i, j in edges] + [(f'-v{i},-v{j}', 1) for i, j in edges]

        res = {'clauses': clauses}

        # Generate piecewise linear function parameters
        num_pieces = self.num_pieces
        breakpoints = sorted(np.random.uniform(self.min_resource, self.max_resource, num_pieces - 1))
        slopes = np.random.uniform(self.min_slope, self.max_slope, num_pieces)
        intercepts = np.random.uniform(self.min_intercept, self.max_intercept, num_pieces)

        res.update({
            'breakpoints': breakpoints,
            'slopes': slopes,
            'intercepts': intercepts
        })

        # Additional data for semi-continuous variables
        res['semi_continuous_bounds'] = (self.min_semi_cont, self.max_semi_cont)

        # New instance data
        demands = self.randint(self.n_customers, self.demand_interval)
        station_capacities = self.randint(self.n_stations, self.station_capacity_interval)
        renewable_capacities = self.renewable_energy_supply()
        fixed_costs = self.randint(self.n_stations, self.fixed_cost_interval)
        transport_costs = self.unit_transportation_costs()

        res.update({
            'demands': demands,
            'station_capacities': station_capacities,
            'renewable_capacities': renewable_capacities,
            'fixed_costs': fixed_costs,
            'transport_costs': transport_costs
        })

        item_weights = self.randint(self.number_of_items, (1, 10))
        item_profits = self.randint(self.number_of_items, (10, 100))
        knapsack_capacities = self.randint(self.number_of_knapsacks, (30, 100))
        
        res.update({
            'item_weights': item_weights,
            'item_profits': item_profits,
            'knapsack_capacities': knapsack_capacities
        })
        
        # New Data for more complex Knapsack Constraints
        new_item_weights = self.randint(self.extra_items, (5, 15))
        knapsack_extra_capacities = self.randint(self.extra_knapsacks, (50, 150))
        renewable_distributions = self.randint(self.additional_renewables, (20, 100))
        
        res.update({
            'new_item_weights': new_item_weights,
            'knapsack_extra_capacities': knapsack_extra_capacities,
            'renewable_distributions': renewable_distributions
        })
        
        return res

    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def unit_transportation_costs(self):
        return np.random.rand(self.n_customers, self.n_stations) * self.transport_cost_scale

    def renewable_energy_supply(self):
        return np.random.rand(self.n_renewables) * self.renewable_capacity_scale

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        clauses = instance['clauses']
        breakpoints = instance['breakpoints']
        slopes = instance['slopes']
        intercepts = instance['intercepts']
        semi_cont_bounds = instance['semi_continuous_bounds']
        demands = instance['demands']
        station_capacities = instance['station_capacities']
        renewable_capacities = instance['renewable_capacities']
        fixed_costs = instance['fixed_costs']
        transport_costs = instance['transport_costs']
        item_weights = instance['item_weights']
        item_profits = instance['item_profits']
        knapsack_capacities = instance['knapsack_capacities']
        new_item_weights = instance['new_item_weights']
        knapsack_extra_capacities = instance['knapsack_extra_capacities']
        renewable_distributions = instance['renewable_distributions']

        model = Model("ComplexSupplyChainAndEVChargingOptimization")

        var_names = {}

        # Create variables for each literal and clause
        for idx, (clause, weight) in enumerate(clauses):
            for var in clause.split(','):
                literal = var[1:] if var.startswith('-') else var
                if literal not in var_names:
                    var_names[literal] = model.addVar(vtype="B", name=literal)
            clause_var = model.addVar(vtype="B", name=f"cl_{idx}")
            var_names[f"cl_{idx}"] = clause_var

        # Semi-continuous variable for resource cost
        min_semi_cont, max_semi_cont = semi_cont_bounds
        semi_resource = model.addVar(vtype="C", lb=min_semi_cont, ub=max_semi_cont, name="semi_resource")

        # Piecewise linear cost function adapted for semi-continuous variable
        semi_cost = model.addVar(vtype="C", name="semi_cost")
        for i in range(len(breakpoints) - 1):
            model.addCons(
                semi_cost >= slopes[i] * semi_resource + intercepts[i],
                name=f"semi_cost_piece_{i}")
        
        model.addCons(
            semi_cost >= slopes[-1] * semi_resource + intercepts[-1],
            name="semi_cost_last_piece")

        # New Variables for EV Charging Station Optimization
        open_stations = {j: model.addVar(vtype="B", name=f"Open_{j}") for j in range(self.n_stations)}
        flow = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}") for i in range(self.n_customers) for j in range(self.n_stations)}
        renewable_supply = {j: model.addVar(vtype="C", name=f"RenewableSupply_{j}") for j in range(self.n_renewables)}
        knapsack_vars = {(i, j): model.addVar(vtype="B", name=f"x_{i}_{j}") for i in range(len(item_weights)) for j in range(len(knapsack_capacities))}
        
        # New Variables for Knapsack Constraints
        new_knapsack_vars = {(i, j): model.addVar(vtype="B", name=f"y_{i}_{j}") for i in range(len(new_item_weights)) for j in range(len(knapsack_extra_capacities))}
        dist_knapsack_vars = {(k, j): model.addVar(vtype="B", name=f"z_{k}_{j}") for k in range(len(renewable_distributions)) for j in range(len(knapsack_extra_capacities))}
        
        # Objective function - maximize the number of satisfied clauses minus semi resource cost
        objective_expr = quicksum(
            var_names[f"cl_{idx}"] * weight for idx, (clause, weight) in enumerate(clauses) if weight < np.inf
        ) - semi_cost

        # Add constraints for each clause
        for idx, (clause, weight) in enumerate(clauses):
            vars_in_clause = clause.split(',')
            clause_var = var_names[f"cl_{idx}"]
            
            # Define the positive and negative parts
            positive_part = quicksum(var_names[var] for var in vars_in_clause if not var.startswith('-'))
            negative_part = quicksum(1 - var_names[var[1:]] for var in vars_in_clause if var.startswith('-'))

            # Total satisfied variables in the clause
            total_satisfied = positive_part + negative_part

            if weight < np.inf:
                model.addCons(total_satisfied >= clause_var, name=f"clause_{idx}")
            else:
                model.addCons(total_satisfied >= 1, name=f"clause_{idx}")

        # Additional Constraints and Objective for EV Charging Station Optimization
        # Include cost terms from the EV charging optimization
        objective_expr -= (
            quicksum(fixed_costs[j] * open_stations[j] for j in range(self.n_stations)) +
            quicksum(transport_costs[i, j] * flow[i, j] for i in range(self.n_customers) for j in range(self.n_stations))
        )

        model.setObjective(objective_expr, "maximize")

        # Demand satisfaction constraints
        for i in range(self.n_customers):
            model.addCons(quicksum(flow[i, j] for j in range(self.n_stations)) == demands[i], f"Demand_{i}")
        
        # Station capacity constraints
        for j in range(self.n_stations):
            model.addCons(quicksum(flow[i, j] for i in range(self.n_customers)) <= station_capacities[j] * open_stations[j], f"StationCapacity_{j}")
        
        # Renewable supply constraints
        for k in range(self.n_renewables):
            model.addCons(renewable_supply[k] <= renewable_capacities[k], f"RenewableCapacity_{k}")

        # Linking renewable supply to station energy inflow
        for j in range(self.n_stations):
            model.addCons(quicksum(renewable_supply[k] for k in range(self.n_renewables)) >= quicksum(flow[i, j] for i in range(self.n_customers)) * open_stations[j], f"RenewableSupplyLink_{j}")

        # Items in at most one knapsack (Set Packing)
        for i in range(len(item_weights)):
            model.addCons(quicksum(knapsack_vars[(i, j)] for j in range(len(knapsack_capacities))) <= 1, f"ItemAssignment_{i}")

        for i in range(len(new_item_weights)):
            model.addCons(quicksum(new_knapsack_vars[(i, j)] for j in range(len(knapsack_extra_capacities))) <= 1, f"NewKnapsackItemAssignment_{i}")

        # Knapsack capacity constraints
        for j in range(len(knapsack_capacities)):
            model.addCons(quicksum(item_weights[i] * knapsack_vars[(i, j)] for i in range(len(item_weights))) <= knapsack_capacities[j], f"KnapsackCapacity_{j}")
            
        for j in range(len(knapsack_extra_capacities)):
            model.addCons(quicksum(new_item_weights[i] * new_knapsack_vars[(i, j)] for i in range(len(new_item_weights))) <= knapsack_extra_capacities[j], f"NewKnapsackCapacity_{j}")
        
        # Each extra knapsack must contain at least one item (Set Covering)
        for j in range(len(knapsack_extra_capacities)):
            model.addCons(quicksum(new_knapsack_vars[(i, j)] for i in range(len(new_item_weights))) >= 1, f"SetCoveringKnapsack_{j}")

        # Renewable distribution constraints 
        for k in range(len(renewable_distributions)):
            model.addCons(quicksum(dist_knapsack_vars[(k, j)] for j in range(len(knapsack_extra_capacities))) == 1, f"RenewableDistribution_{k}")

        # Logical conditions added to enhance complexity
        # Logical Condition 1: If a specific station is open, a specific item must be in a knapsack
        specific_station, specific_item = 0, 2
        model.addCons(open_stations[specific_station] == knapsack_vars[(specific_item, 0)], f"LogicalCondition_ItemPlacement_{specific_item}_{specific_station}")

        # Logical Condition 2: If certain stations are used, renewable supply must be linked logically
        for j in range(self.n_stations):
            model.addCons(quicksum(flow[i, j] for i in range(self.n_customers)) * open_stations[j] <= quicksum(renewable_supply[k] for k in range(self.n_renewables)), f"LogicalCondition_StationRenewable_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 0,
        'max_n': 280,
        'er_prob': 0.59,
        'edge_addition_prob': 0.31,
        'num_pieces': 13,
        'min_resource': 0,
        'max_resource': 3,
        'min_slope': 0,
        'max_slope': 9,
        'min_intercept': 0,
        'max_intercept': 0,
        'min_semi_cont': 0,
        'max_semi_cont': 420,
        'n_customers': 37,
        'n_stations': 300,
        'n_renewables': 30,
        'demand_interval': (210, 1050),
        'station_capacity_interval': (350, 1400),
        'renewable_capacity_scale': 2250.0,
        'fixed_cost_interval': (26, 105),
        'transport_cost_scale': 810.0,
        'number_of_items': 250,
        'number_of_knapsacks': 20,
        'extra_items': 300,
        'extra_knapsacks': 7,
        'additional_renewables': 10,
    }

    optimizer = ComplexSupplyChainAndEVChargingOptimization(parameters, seed=seed)
    instance = optimizer.generate_instances()
    solve_status, solve_time = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")