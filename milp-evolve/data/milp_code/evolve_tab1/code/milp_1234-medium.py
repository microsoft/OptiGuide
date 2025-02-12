import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SupplyChainNetworkDesign:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_supply_chain_data(self, F, W, R):
        capacities_factory = np.random.randint(100, 500, size=F)
        capacities_warehouse = np.random.randint(200, 1000, size=W)
        demands_retailer = np.random.randint(50, 300, size=R)
        fixed_costs_warehouse = np.random.randint(1000, 5000, size=W)
        trans_cost_fw = np.random.randint(5, 20, size=(F, W))
        trans_cost_wr = np.random.randint(5, 20, size=(W, R))

        res = {
            'capacities_factory': capacities_factory,
            'capacities_warehouse': capacities_warehouse,
            'demands_retailer': demands_retailer,
            'fixed_costs_warehouse': fixed_costs_warehouse,
            'trans_cost_fw': trans_cost_fw,
            'trans_cost_wr': trans_cost_wr,
        }
        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        capacities_factory = instance['capacities_factory']
        capacities_warehouse = instance['capacities_warehouse']
        demands_retailer = instance['demands_retailer']
        fixed_costs_warehouse = instance['fixed_costs_warehouse']
        trans_cost_fw = instance['trans_cost_fw']
        trans_cost_wr = instance['trans_cost_wr']

        num_factories = len(capacities_factory)
        num_warehouses = len(capacities_warehouse)
        num_retailers = len(demands_retailer)

        model = Model("SupplyChainNetworkDesign")

        # Create variables for the shipment flows and warehouse openings
        Factory_to_Warehouse_flow = {}
        Warehouse_to_Retailer_flow = {}
        Open_Warehouse = {}

        for f in range(num_factories):
            for w in range(num_warehouses):
                Factory_to_Warehouse_flow[f, w] = model.addVar(vtype="C", name=f"flow_F{f}_W{w}")
        
        for w in range(num_warehouses):
            for r in range(num_retailers):
                Warehouse_to_Retailer_flow[w, r] = model.addVar(vtype="C", name=f"flow_W{w}_R{r}")

            Open_Warehouse[w] = model.addVar(vtype="B", name=f"open_W{w}")

        # Objective function: minimize total cost
        fixed_cost_term = quicksum(Open_Warehouse[w] * fixed_costs_warehouse[w] for w in range(num_warehouses))
        trans_cost_term_fw = quicksum(Factory_to_Warehouse_flow[f, w] * trans_cost_fw[f, w] for f in range(num_factories) for w in range(num_warehouses))
        trans_cost_term_wr = quicksum(Warehouse_to_Retailer_flow[w, r] * trans_cost_wr[w, r] for w in range(num_warehouses) for r in range(num_retailers))
        
        model.setObjective(fixed_cost_term + trans_cost_term_fw + trans_cost_term_wr, "minimize")

        # Constraints
        # Fulfill demand
        for r in range(num_retailers):
            model.addCons(quicksum(Warehouse_to_Retailer_flow[w, r] for w in range(num_warehouses)) == demands_retailer[r], 
                          name=f"Fulfill_Demand_R{r}")

        # Capacity of warehouses
        for w in range(num_warehouses):
            model.addCons(quicksum(Warehouse_to_Retailer_flow[w, r] for r in range(num_retailers)) <= quicksum(Factory_to_Warehouse_flow[f, w] for f in range(num_factories)), 
                          name=f"Capacity_Constraint_W{w}")

        # Capacity of factories
        for f in range(num_factories):
            model.addCons(quicksum(Factory_to_Warehouse_flow[f, w] for w in range(num_warehouses)) <= capacities_factory[f], 
                          name=f"Capacity_Constraint_F{f}")

        # Flow conservation at warehouses
        for w in range(num_warehouses):
            model.addCons(quicksum(Factory_to_Warehouse_flow[f, w] for f in range(num_factories)) <= capacities_warehouse[w], 
                          name=f"Flow_Conservation_W{w}")

        # Warehouse opening constraint
        for w in range(num_warehouses):
            for r in range(num_retailers):
                model.addCons(Warehouse_to_Retailer_flow[w, r] <= Open_Warehouse[w] * demands_retailer[r], 
                              name=f"Open_Warehouse_Constraint_W{w}_R{r}")


        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_factories': 175,
        'num_warehouses': 80,
        'num_retailers': 35,
    }

    scnd = SupplyChainNetworkDesign(parameters, seed=seed)
    instance = scnd.generate_supply_chain_data(parameters['num_factories'], parameters['num_warehouses'], parameters['num_retailers'])
    solve_status, solve_time = scnd.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")