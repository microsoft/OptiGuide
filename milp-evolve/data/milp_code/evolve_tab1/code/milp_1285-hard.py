import random
import time
import numpy as np
from itertools import product
from pyscipopt import Model, quicksum

class SupplyChainNetwork:
    def __init__(self, number_of_factories, number_of_warehouses, number_of_customers, factory_costs, warehouse_costs, transport_costs, factory_capacities, customer_demands):
        self.number_of_factories = number_of_factories
        self.number_of_warehouses = number_of_warehouses
        self.number_of_customers = number_of_customers
        self.factory_costs = factory_costs
        self.warehouse_costs = warehouse_costs
        self.transport_costs = transport_costs
        self.factory_capacities = factory_capacities
        self.customer_demands = customer_demands

class SupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed) 
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_factories > 0 and self.n_warehouses > 0 and self.n_customers > 0
        assert self.min_factory_cost >= 0 and self.max_factory_cost >= self.min_factory_cost
        assert self.min_warehouse_cost >= 0 and self.max_warehouse_cost >= self.min_warehouse_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_factory_capacity > 0 and self.max_factory_capacity >= self.min_factory_capacity
        assert self.min_customer_demand > 0 and self.max_customer_demand >= self.min_customer_demand

        factory_costs = np.random.randint(self.min_factory_cost, self.max_factory_cost + 1, self.n_factories)
        warehouse_costs = np.random.randint(self.min_warehouse_cost, self.max_warehouse_cost + 1, self.n_warehouses)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_factories, self.n_warehouses, self.n_customers))
        factory_capacities = np.random.randint(self.min_factory_capacity, self.max_factory_capacity + 1, self.n_factories)
        customer_demands = np.random.randint(self.min_customer_demand, self.max_customer_demand + 1, self.n_customers)
        
        return {
            "factory_costs": factory_costs,
            "warehouse_costs": warehouse_costs,
            "transport_costs": transport_costs,
            "factory_capacities": factory_capacities,
            "customer_demands": customer_demands
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        factory_costs = instance['factory_costs']
        warehouse_costs = instance['warehouse_costs']
        transport_costs = instance['transport_costs']
        factory_capacities = instance['factory_capacities']
        customer_demands = instance['customer_demands']
        
        model = Model("SupplyChainOptimization")
        n_factories = len(factory_costs)
        n_warehouses = len(warehouse_costs)
        n_customers = len(customer_demands)
        
        # Decision variables
        factory_vars = {f: model.addVar(vtype="B", name=f"Factory_{f}") for f in range(n_factories)}
        warehouse_vars = {(f, w): model.addVar(vtype="B", name=f"Factory_{f}_Warehouse_{w}") for f in range(n_factories) for w in range(n_warehouses)}
        transport_vars = {(f, w, c): model.addVar(vtype="C", name=f"Factory_{f}_Warehouse_{w}_Customer_{c}") for f in range(n_factories) for w in range(n_warehouses) for c in range(n_customers)}

        # Objective: minimize the total cost including factory, warehouse, and transport costs
        model.setObjective(
            quicksum(factory_costs[f] * factory_vars[f] for f in range(n_factories)) +
            quicksum(warehouse_costs[w] * warehouse_vars[f, w] for f in range(n_factories) for w in range(n_warehouses)) +
            quicksum(transport_costs[f, w, c] * transport_vars[f, w, c] for f in range(n_factories) for w in range(n_warehouses) for c in range(n_customers)), 
            "minimize"
        )
        
        # Constraints: Each customer demand is met
        for c in range(n_customers):
            model.addCons(quicksum(transport_vars[f, w, c] for f in range(n_factories) for w in range(n_warehouses)) == customer_demands[c], f"Customer_{c}_Demand")
        
        # Constraints: Factories and Warehouses capacities are not exceeded
        for f in range(n_factories):
            model.addCons(quicksum(transport_vars[f, w, c] for w in range(n_warehouses) for c in range(n_customers)) <= factory_capacities[f], f"Factory_{f}_Capacity")
        
        for f in range(n_factories):
            for w in range(n_warehouses):
                model.addCons(quicksum(transport_vars[f, w, c] for c in range(n_customers)) <= self.max_warehouse_capacity * warehouse_vars[f, w], f"Warehouse_{w}_Capacity_By_Factory_{f}")
        
        # Big M Formulation constraint: Only open factories can serve warehouses
        for f in range(n_factories):
            for w in range(n_warehouses):
                model.addCons(warehouse_vars[f, w] <= factory_vars[f], f"Factory_{f}_Opens_Warehouse_{w}")
        
        # Constraint: Each warehouse can be linked to only one factory
        for w in range(n_warehouses):
            model.addCons(quicksum(warehouse_vars[f, w] for f in range(n_factories)) == 1, f"Warehouse_{w}_Link")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_factories': 30,
        'n_warehouses': 30,
        'n_customers': 7,
        'min_factory_cost': 3000,
        'max_factory_cost': 5000,
        'min_warehouse_cost': 1000,
        'max_warehouse_cost': 3000,
        'min_transport_cost': 21,
        'max_transport_cost': 87,
        'min_factory_capacity': 600,
        'max_factory_capacity': 1000,
        'min_customer_demand': 112,
        'max_customer_demand': 2700,
        'max_warehouse_capacity': 3000,
    }

    supply_chain_optimizer = SupplyChainOptimization(parameters, seed=42)
    instance = supply_chain_optimizer.generate_instance()
    solve_status, solve_time, objective_value = supply_chain_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")