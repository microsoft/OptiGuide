import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class MultipleKnapsackWithBidsAndSupplyNetwork:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        # Original instance generation
        weights = np.random.randint(self.min_range, self.max_range, self.number_of_items)

        if self.scheme == 'uncorrelated':
            profits = np.random.randint(self.min_range, self.max_range, self.number_of_items)
        elif self.scheme == 'weakly correlated':
            profits = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(weights - (self.max_range - self.min_range), 1),
                    weights + (self.max_range - self.min_range)]))
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

        n_bids = int(self.number_of_items / 3)
        bids = []
        for _ in range(n_bids):
            selected_items = np.random.choice(self.number_of_items, self.items_per_bid, replace=False)
            bid_profit = profits[selected_items].sum()
            bids.append((selected_items.tolist(), bid_profit))
        
        bid_pairs = []
        for _ in range(n_bids // 4):
            bid_pairs.append((
                random.randint(0, n_bids - 1),
                random.randint(0, n_bids - 1)
            ))

        # Additional Data for Supply Chain Problem
        num_manufacturing_hubs = self.number_of_knapsacks  # Treat knapsacks as manufacturing hubs
        num_distribution_centers = self.number_of_items  # Treat each item as a distribution center

        transportation_costs = np.random.randint(50, 300, size=(num_distribution_centers, num_manufacturing_hubs))
        fixed_costs = np.random.randint(1000, 5000, size=num_manufacturing_hubs)
        distribution_demand = np.random.randint(1, 10, size=num_distribution_centers)
        
        res = {
            'weights': weights,
            'profits': profits,
            'capacities': capacities,
            'bids': bids,
            'mutual_exclusivity_pairs': bid_pairs,
            'num_manufacturing_hubs': num_manufacturing_hubs,
            'num_distribution_centers': num_distribution_centers,
            'transportation_costs': transportation_costs,
            'fixed_costs': fixed_costs,
            'distribution_demand': distribution_demand,
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        weights = instance['weights']
        profits = instance['profits']
        capacities = instance['capacities']
        bids = instance['bids']
        mutual_exclusivity_pairs = instance['mutual_exclusivity_pairs']
        
        num_manufacturing_hubs = instance['num_manufacturing_hubs']
        num_distribution_centers = instance['num_distribution_centers']
        transportation_costs = instance['transportation_costs']
        fixed_costs = instance['fixed_costs']
        distribution_demand = instance['distribution_demand']
        
        number_of_items = len(weights)
        number_of_knapsacks = len(capacities)

        model = Model("MultipleKnapsackWithBidsAndSupplyNetwork")
        
        var_names = {}
        x_bids = {i: model.addVar(vtype="B", name=f"x_bid_{i}") for i in range(len(bids))}
        waste_vars = {i: model.addVar(vtype="C", name=f"waste_{i}") for i in range(len(bids))}
        manufacturing_hub = {j: model.addVar(vtype="B", name=f"manufacturing_hub_{j}") for j in range(num_manufacturing_hubs)}
        transportation = {(i, j): model.addVar(vtype="B", name=f"transportation_{i}_{j}") for i in range(num_distribution_centers) for j in range(num_manufacturing_hubs)}

        # Decision variables: x[i][j] = 1 if item i is placed in knapsack j
        for i in range(number_of_items):
            for j in range(number_of_knapsacks):
                var_names[(i, j)] = model.addVar(vtype="B", name=f"x_{i}_{j}")

        # Objective: Maximize total profit and minimize total cost
        objective_expr = quicksum(profits[i] * var_names[(i, j)] for i in range(number_of_items) for j in range(number_of_knapsacks)) \
                         - quicksum(waste_vars[i] for i in range(len(bids))) \
                         - quicksum(transportation[i, j] * transportation_costs[i, j] for i in range(num_distribution_centers) for j in range(num_manufacturing_hubs)) \
                         - quicksum(manufacturing_hub[j] * fixed_costs[j] for j in range(num_manufacturing_hubs))

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

        # Constraints: Handle the newly introduced bids
        for i, (selected_items, bid_profit) in enumerate(bids):
            model.addCons(
                quicksum(var_names[(item, k)] for item in selected_items for k in range(number_of_knapsacks)) <= len(selected_items) * x_bids[i]
            )
        
        # Constraints: Mutual exclusivity among certain bids
        for (bid1, bid2) in mutual_exclusivity_pairs:
            model.addCons(x_bids[bid1] + x_bids[bid2] <= 1, name=f"Exclusive_bids_{bid1}_{bid2}")

        # Constraints: Waste penalty
        for i in range(len(bids)):
            model.addCons(waste_vars[i] >= 0, f"Waste_LB_{i}")
            model.addCons(waste_vars[i] >= self.waste_factor * (1 - x_bids[i]), f"Waste_Link_{i}")

        # New Constraints from Supply Chain Problem

        # Set-covering constraints: Each distribution center must be covered by at least one knapsack (hub)
        for i in range(num_distribution_centers):
            model.addCons(quicksum(transportation[i, j] for j in range(num_manufacturing_hubs)) >= 1, name=f"distribution_coverage_{i}")

        # Logical constraints: A distribution center can only be supplied if the knapsack (hub) is operational
        for j in range(num_manufacturing_hubs):
            for i in range(num_distribution_centers):
                model.addCons(transportation[i, j] <= manufacturing_hub[j], name=f"manufacturing_hub_connection_{i}_{j}")

        # Capacity constraints: A knapsack (hub) can only supply up to its capacity
        for j in range(num_manufacturing_hubs):
            model.addCons(quicksum(transportation[i, j] * distribution_demand[i] for i in range(num_distribution_centers)) <= capacities[j], name=f"manufacturing_capacity_{j}")

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_items': 500,
        'number_of_knapsacks': 7,
        'min_range': 4,
        'max_range': 900,
        'scheme': 'weakly correlated',
        'items_per_bid': 14,
        'waste_factor': 0.52,
        'min_manufacturing_hubs': 5,
        'max_manufacturing_hubs': 175,
        'min_distribution_centers': 45,
        'max_distribution_centers': 1120,
    }

    knapsack_with_network = MultipleKnapsackWithBidsAndSupplyNetwork(parameters, seed=seed)
    instance = knapsack_with_network.generate_instance()
    solve_status, solve_time = knapsack_with_network.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")