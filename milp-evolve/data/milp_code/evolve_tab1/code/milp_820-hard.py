import random
import time
import numpy as np
from scipy.stats import poisson, binom
from pyscipopt import Model, quicksum

class EnhancedCombinatorialAuctionWithFactoryAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        assert self.min_value >= 0 and self.max_value >= self.min_value
        assert self.add_item_prob >= 0 and self.add_item_prob <= 1

        # Common item values (resale price)
        values = self.min_value + (self.max_value - self.min_value) * np.random.rand(self.n_items)

        # Bids generation
        bids = []
        for _ in range(self.n_bids):
            private_interests = np.random.rand(self.n_items)
            private_values = values + self.max_value * self.value_deviation * (2 * private_interests - 1)

            initial_item = np.random.choice(self.n_items, p=private_interests / private_interests.sum())
            bundle_mask = np.zeros(self.n_items, dtype=bool)
            bundle_mask[initial_item] = True

            while np.random.rand() < self.add_item_prob and bundle_mask.sum() < self.n_items:
                next_item = np.random.choice(self.n_items, p=(private_interests * ~bundle_mask) / (private_interests * ~bundle_mask).sum())
                bundle_mask[next_item] = True

            bundle = np.nonzero(bundle_mask)[0]
            price = private_values[bundle].sum() + len(bundle) ** (1 + self.additivity)

            if price > 0:
                bids.append((list(bundle), price))

        # Capacity constraints and activation costs
        transportation_capacity = poisson.rvs(mu=10, size=self.n_items)
        activation_costs = np.random.normal(loc=100, scale=20, size=len(bids))

        # Bounds for new semi-continuous variables
        semi_continuous_bounds = np.random.uniform(50, 150, size=len(bids))

        # Factory capacities and operational costs data
        factory_capacities = self.min_capacity + (self.max_capacity - self.min_capacity) * np.random.rand(self.n_factories)
        factory_costs = np.random.exponential(50, size=self.n_factories).tolist()

        # New data: Bid dependencies and processing limitations
        bid_processing_times = np.random.randint(1, 10, size=len(bids))
        factory_processing_limits = np.random.randint(50, 200, size=self.n_factories)
        bid_dependencies = [(i, j) for i in range(len(bids)) for j in range(i+1, len(bids)) if np.random.rand() < 0.1]

        bids_per_item = [[] for _ in range(self.n_items)]
        for i, bid in enumerate(bids):
            bundle, price = bid
            for item in bundle:
                bids_per_item[item].append(i)

        return {
            "bids": bids,
            "bids_per_item": bids_per_item,
            "transportation_capacity": transportation_capacity,
            "activation_costs": activation_costs,
            "semi_continuous_bounds": semi_continuous_bounds,
            "factory_capacities": factory_capacities,
            "factory_costs": factory_costs,
            "bid_processing_times": bid_processing_times,
            "factory_processing_limits": factory_processing_limits,
            "bid_dependencies": bid_dependencies
        }

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        bids = instance['bids']
        bids_per_item = instance['bids_per_item']
        transportation_capacity = instance['transportation_capacity']
        activation_costs = instance['activation_costs']
        semi_continuous_bounds = instance['semi_continuous_bounds']
        factory_capacities = instance['factory_capacities']
        factory_costs = instance['factory_costs']
        bid_processing_times = instance['bid_processing_times']
        factory_processing_limits = instance['factory_processing_limits']
        bid_dependencies = instance['bid_dependencies']
        
        model = Model("EnhancedCombinatorialAuctionWithFactoryAllocation")
        
        # Decision variables
        bid_vars = {i: model.addVar(vtype="C", lb=0, ub=semi_continuous_bounds[i], name=f"Bid_{i}") for i in range(len(bids))}
        activate_vars = {i: model.addVar(vtype="I", lb=0, ub=1, name=f"Activate_{i}") for i in range(len(bids))}
        factory_usage_vars = {i: model.addVar(vtype="B", name=f"FactoryUsage_{i}") for i in range(self.n_factories)}
        
        # Objective: maximize the total price minus activation costs and factory operational costs
        objective_expr = (
            quicksum(price * bid_vars[i] - activation_costs[i] * activate_vars[i] for i, (bundle, price) in enumerate(bids)) -
            quicksum(factory_costs[i] * factory_usage_vars[i] for i in range(self.n_factories))
        )
        
        ### Constraints ###
        # Each item can be in at most one bundle
        for item, bid_indices in enumerate(bids_per_item):
            model.addCons(quicksum(bid_vars[bid_idx] for bid_idx in bid_indices) <= transportation_capacity[item], f"Item_{item}")
        
        # Ensure activation variable is set if any bids for the bundle is accepted
        for i in range(len(bids)):
            model.addCons(bid_vars[i] >= semi_continuous_bounds[i] * activate_vars[i], f"Activation_Bid_{i}")
        
        # Factory capacity constraints
        for i in range(self.n_factories):
            model.addCons(
                quicksum(bid_vars[bid_idx] for bid_idx, (bundle, price) in enumerate(bids) if i in bundle) <= factory_capacities[i] * factory_usage_vars[i],
                f"FactoryCapacity_{i}"
            )

        # New constraint: Factory processing time limits
        for i in range(self.n_factories):
            model.addCons(
                quicksum(bid_vars[bid_idx] * bid_processing_times[bid_idx] for bid_idx, (bundle, price) in enumerate(bids) if i in bundle) <= factory_processing_limits[i],
                f"FactoryProcessingLimit_{i}"
            )
            
        # New constraint: Bid dependencies (one bid requires another to be processed first)
        for i, j in bid_dependencies:
            model.addCons(activate_vars[i] <= activate_vars[j], f"BidDependency_{i}_{j}")

        model.setObjective(objective_expr, "maximize")
                
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_items': 2500,
        'n_bids': 1518,
        'min_value': 250,
        'max_value': 1800,
        'value_deviation': 0.8,
        'additivity': 0.67,
        'add_item_prob': 0.79,
        'min_teams': 245,
        'max_teams': 3000,
        'min_locations': 6,
        'max_locations': 1400,
        'n_factories': 1200,
        'min_capacity': 1800,
        'max_capacity': 3000,
        'new_param_1': 350,
        'new_param_2': 1200,
    }
    
    auction = EnhancedCombinatorialAuctionWithFactoryAllocation(parameters, seed=seed)
    instance = auction.generate_instance()
    solve_status, solve_time = auction.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")