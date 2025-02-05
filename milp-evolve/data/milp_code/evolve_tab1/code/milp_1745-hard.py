import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_suppliers > 0 and self.n_dcs > 0 and self.n_retailers > 0
        assert self.min_supply_cost >= 0 and self.max_supply_cost >= self.min_supply_cost
        assert self.min_trans_cost >= 0 and self.max_trans_cost >= self.min_trans_cost
        assert self.min_supply_capacity > 0 and self.max_supply_capacity >= self.min_supply_capacity
        assert self.min_shed_penalty >= 0 and self.max_shed_penalty >= self.min_shed_penalty

        supply_costs = np.random.randint(self.min_supply_cost, self.max_supply_cost + 1, self.n_suppliers)
        trans_costs_sd = np.random.randint(self.min_trans_cost, self.max_trans_cost + 1, (self.n_suppliers, self.n_dcs))
        trans_costs_dr = np.random.randint(self.min_trans_cost, self.max_trans_cost + 1, (self.n_dcs, self.n_retailers))
        supply_capacities = np.random.randint(self.min_supply_capacity, self.max_supply_capacity + 1, self.n_suppliers)
        dc_capacities = np.random.randint(self.min_dc_capacity, self.max_dc_capacity + 1, self.n_dcs)
        retailer_demands = np.random.randint(1, 10, self.n_retailers)
        shed_penalties = np.random.uniform(self.min_shed_penalty, self.max_shed_penalty, self.n_retailers)

        return {
            "supply_costs": supply_costs,
            "trans_costs_sd": trans_costs_sd,
            "trans_costs_dr": trans_costs_dr,
            "supply_capacities": supply_capacities,
            "dc_capacities": dc_capacities,
            "retailer_demands": retailer_demands,
            "shed_penalties": shed_penalties,
        }

    def solve(self, instance):
        supply_costs = instance['supply_costs']
        trans_costs_sd = instance['trans_costs_sd']
        trans_costs_dr = instance['trans_costs_dr']
        supply_capacities = instance['supply_capacities']
        dc_capacities = instance['dc_capacities']
        retailer_demands = instance['retailer_demands']
        shed_penalties = instance['shed_penalties']

        model = Model("SupplyChainOptimization")
        n_suppliers = len(supply_costs)
        n_dcs = len(trans_costs_sd[0])
        n_retailers = len(trans_costs_dr[0])

        # Decision variables
        supplier_vars = {s: model.addVar(vtype="B", name=f"Supplier_{s}") for s in range(n_suppliers)}
        flow_sd_vars = {(s, d): model.addVar(vtype="C", name=f"Flow_SD_{s}_{d}") for s in range(n_suppliers) for d in range(n_dcs)}
        flow_dr_vars = {(d, r): model.addVar(vtype="C", name=f"Flow_DR_{d}_{r}") for d in range(n_dcs) for r in range(n_retailers)}
        shed_vars = {r: model.addVar(vtype="C", name=f"Shed_{r}") for r in range(n_retailers)}

        # Objective: minimize the total cost
        model.setObjective(
            quicksum(supply_costs[s] * supplier_vars[s] for s in range(n_suppliers)) +
            quicksum(trans_costs_sd[s, d] * flow_sd_vars[(s, d)] for s in range(n_suppliers) for d in range(n_dcs)) +
            quicksum(trans_costs_dr[d, r] * flow_dr_vars[(d, r)] for d in range(n_dcs) for r in range(n_retailers)) +
            quicksum(shed_penalties[r] * shed_vars[r] for r in range(n_retailers)),
            "minimize"
        )

        # Flow conservation constraint at each DC
        for d in range(n_dcs):
            model.addCons(
                quicksum(flow_sd_vars[(s, d)] for s in range(n_suppliers)) ==
                quicksum(flow_dr_vars[(d, r)] for r in range(n_retailers)),
                f"FlowConservation_DC_{d}"
            )

        # Capacity constraints for suppliers
        for s in range(n_suppliers):
            model.addCons(
                quicksum(flow_sd_vars[(s, d)] for d in range(n_dcs)) <= supply_capacities[s] * supplier_vars[s], 
                f"Supplier_{s}_Capacity"
            )

        # Capacity constraints for DCs
        for d in range(n_dcs):
            model.addCons(
                quicksum(flow_dr_vars[(d, r)] for r in range(n_retailers)) <= dc_capacities[d], 
                f"DC_{d}_Capacity"
            )

        # Demand satisfaction constraints at each retailer
        for r in range(n_retailers):
            model.addCons(
                quicksum(flow_dr_vars[(d, r)] for d in range(n_dcs)) + shed_vars[r] == retailer_demands[r], 
                f"Demand_Retailer_{r}"
            )

        # Non-negativity constraints for shed variables
        for r in range(n_retailers):
            model.addCons(shed_vars[r] >= 0, f"Shed_{r}_NonNegative")

        # Start optimization
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_suppliers': 350,
        'n_dcs': 22,
        'n_retailers': 600,
        'min_trans_cost': 25,
        'max_trans_cost': 180,
        'min_supply_cost': 270,
        'max_supply_cost': 560,
        'min_supply_capacity': 100,
        'max_supply_capacity': 1200,
        'min_dc_capacity': 2250,
        'max_dc_capacity': 2500,
        'min_shed_penalty': 7,
        'max_shed_penalty': 750,
    }

    optimizer = SupplyChainOptimization(parameters, seed=seed)
    instance = optimizer.generate_instance()
    solve_status, solve_time, objective_value = optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")