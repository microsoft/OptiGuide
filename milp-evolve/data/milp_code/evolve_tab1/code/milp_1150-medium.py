import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FactoryExpansionOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.NumberOfFactories > 0 and self.GoodsPerDistributionCenter > 0
        assert self.OperationalCostRange[0] >= 0 and self.OperationalCostRange[1] >= self.OperationalCostRange[0]
        assert self.FactoryCapacityRange[0] > 0 and self.FactoryCapacityRange[1] >= self.FactoryCapacityRange[0]

        operational_costs = np.random.randint(self.OperationalCostRange[0], self.OperationalCostRange[1] + 1, self.NumberOfFactories)
        transit_costs = np.random.randint(self.OperationalCostRange[0], self.OperationalCostRange[1] + 1, (self.NumberOfFactories, self.GoodsPerDistributionCenter))
        capacities = np.random.randint(self.FactoryCapacityRange[0], self.FactoryCapacityRange[1] + 1, self.NumberOfFactories)
        goods_demands = np.random.randint(self.GoodsDemandRange[0], self.GoodsDemandRange[1] + 1, self.GoodsPerDistributionCenter)
        no_delivery_penalties = np.random.uniform(100, 300, self.GoodsPerDistributionCenter).tolist()

        critical_centers_subsets = [random.sample(range(self.GoodsPerDistributionCenter), int(0.2 * self.GoodsPerDistributionCenter)) for _ in range(5)]
        min_coverage = np.random.randint(1, 5, 5)

        return {
            "operational_costs": operational_costs,
            "transit_costs": transit_costs,
            "capacities": capacities,
            "goods_demands": goods_demands,
            "no_delivery_penalties": no_delivery_penalties,
            "critical_centers_subsets": critical_centers_subsets,
            "min_coverage": min_coverage
        }

    def solve(self, instance):
        operational_costs = instance['operational_costs']
        transit_costs = instance['transit_costs']
        capacities = instance['capacities']
        goods_demands = instance['goods_demands']
        no_delivery_penalties = instance['no_delivery_penalties']
        critical_centers_subsets = instance['critical_centers_subsets']
        min_coverage = instance['min_coverage']

        model = Model("FactoryExpansionOptimization")
        n_factories = len(operational_costs)
        n_goods = len(goods_demands)

        factory_vars = {f: model.addVar(vtype="B", name=f"Factory_{f}") for f in range(n_factories)}
        goods_assignment_vars = {(f, g): model.addVar(vtype="C", name=f"Factory_{f}_Goods_{g}") for f in range(n_factories) for g in range(n_goods)}
        unhandled_goods_vars = {g: model.addVar(vtype="C", name=f"Unhandled_Goods_{g}") for g in range(n_goods)}

        model.setObjective(
            quicksum(operational_costs[f] * factory_vars[f] for f in range(n_factories)) +
            quicksum(transit_costs[f][g] * goods_assignment_vars[f, g] for f in range(n_factories) for g in range(n_goods)) +
            quicksum(no_delivery_penalties[g] * unhandled_goods_vars[g] for g in range(n_goods)),
            "minimize"
        )

        # Constraints
        # Goods demand satisfaction (total deliveries and unhandled goods must cover total demand)
        for g in range(n_goods):
            model.addCons(quicksum(goods_assignment_vars[f, g] for f in range(n_factories)) + unhandled_goods_vars[g] == goods_demands[g], f"Goods_Demand_Satisfaction_{g}")

        # Capacity limits for each factory
        for f in range(n_factories):
            model.addCons(quicksum(goods_assignment_vars[f, g] for g in range(n_goods)) <= capacities[f] * factory_vars[f], f"Factory_Capacity_{f}")

        # Goods can be assigned only if factory is operational
        for f in range(n_factories):
            for g in range(n_goods):
                model.addCons(goods_assignment_vars[f, g] <= goods_demands[g] * factory_vars[f], f"Operational_Constraint_{f}_{g}")

        # Ensure minimum number of goods handled by some factories for critical distribution centers
        for i, subset in enumerate(critical_centers_subsets):
            model.addCons(quicksum(goods_assignment_vars[f, g] for f in range(n_factories) for g in subset) >= min_coverage[i], f"Set_Covering_Constraint_{i}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'NumberOfFactories': 1000,
        'GoodsPerDistributionCenter': 60,
        'OperationalCostRange': (900, 1200),
        'FactoryCapacityRange': (1125, 1350),
        'GoodsDemandRange': (250, 1250),
    }

    factory_optimizer = FactoryExpansionOptimization(parameters, seed=42)
    instance = factory_optimizer.generate_instance()
    solve_status, solve_time, objective_value = factory_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")