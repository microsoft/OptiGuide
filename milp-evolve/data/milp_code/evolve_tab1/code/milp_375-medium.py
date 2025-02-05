import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class FactoryProduction:
    
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_machine_capacities(self):
        return np.random.randint(100, 500, size=self.num_machines)
    
    def generate_production_costs(self):
        return np.random.uniform(10, 50, size=(self.num_machines, self.num_products))
    
    def generate_setup_costs(self):
        return np.random.uniform(5, 10, size=(self.num_products, self.num_products))
    
    def generate_customer_demand(self):
        return np.random.randint(200, 1000, size=self.num_products)
    
    def generate_vehicle_capacities(self):
        return np.random.randint(500, 1000, size=self.num_vehicles)
    
    def generate_vehicle_distances(self):
        return np.random.uniform(20, 100, size=(self.num_vehicles, self.num_nodes, self.num_nodes))
    
    def generate_node_weights(self):
        return np.random.randint(50, 200, size=self.num_nodes)
    
    def get_instance(self):
        machine_capacities = self.generate_machine_capacities()
        production_costs = self.generate_production_costs()
        setup_costs = self.generate_setup_costs()
        customer_demand = self.generate_customer_demand()
        vehicle_capacities = self.generate_vehicle_capacities()
        vehicle_distances = self.generate_vehicle_distances()
        node_weights = self.generate_node_weights()
        
        return {
            'machine_capacities': machine_capacities,
            'production_costs': production_costs,
            'setup_costs': setup_costs,
            'customer_demand': customer_demand,
            'vehicle_capacities': vehicle_capacities,
            'vehicle_distances': vehicle_distances,
            'node_weights': node_weights,
        }
    
    def solve(self, instance):
        machine_capacities = instance['machine_capacities']
        production_costs = instance['production_costs']
        setup_costs = instance['setup_costs']
        customer_demand = instance['customer_demand']
        vehicle_capacities = instance['vehicle_capacities']
        vehicle_distances = instance['vehicle_distances']
        node_weights = instance['node_weights']
        
        model = Model("FactoryProduction")
        
        # Variables
        product_production_vars = {
            (m, p, t): model.addVar(vtype="C", name=f"Prod_{m}_{p}_{t}")
            for m in range(self.num_machines)
            for p in range(self.num_products)
            for t in range(self.num_periods)
        }
        
        setup_vars = {
            (m, p, t): model.addVar(vtype="B", name=f"Setup_{m}_{p}_{t}")
            for m in range(self.num_machines)
            for p in range(self.num_products)
            for t in range(self.num_periods)
        }
        
        load_vars = {
            (v, n): model.addVar(vtype="C", name=f"Load_{v}_{n}")
            for v in range(self.num_vehicles)
            for n in range(self.num_nodes)
        }
        
        distance_traveled_vars = {
            (v, i, j): model.addVar(vtype="B", name=f"Distance_{v}_{i}_{j}")
            for v in range(self.num_vehicles)
            for i in range(self.num_nodes)
            for j in range(self.num_nodes)
        }
        
        # Objective
        objective_expr = quicksum(
            self.product_prices[p] * product_production_vars[m, p, t]
            for m in range(self.num_machines)
            for p in range(self.num_products)
            for t in range(self.num_periods)
        )
        objective_expr -= quicksum(
            production_costs[m][p] * product_production_vars[m, p, t]
            for m in range(self.num_machines)
            for p in range(self.num_products)
            for t in range(self.num_periods)
        )
        objective_expr -= quicksum(
            setup_costs[p1][p2] * setup_vars[m, p1, t]
            for m in range(self.num_machines)
            for p1 in range(self.num_products)
            for p2 in range(self.num_products)
            for t in range(self.num_periods)
        )
        
        model.setObjective(objective_expr, "maximize")
        
        # Constraints
        for p in range(self.num_products):
            model.addCons(
                quicksum(product_production_vars[m, p, t]
                         for m in range(self.num_machines)
                         for t in range(self.num_periods)) >= customer_demand[p],
                         name=f"Demand_{p}"
            )
        
        for m in range(self.num_machines):
            for t in range(self.num_periods):
                model.addCons(
                    quicksum(product_production_vars[m, p, t] for p in range(self.num_products)) <= machine_capacities[m],
                    name=f"MachineCapacity_{m}_{t}"
                )
        
        for m in range(self.num_machines):
            for t in range(1, self.num_periods):
                for p in range(self.num_products):
                    model.addCons(
                        setup_vars[m, p, t] >= (
                            quicksum(product_production_vars[m, p_old, t-1] for p_old in range(self.num_products)) -
                            product_production_vars[m, p, t]
                        ),
                        name=f"SetupTransition_{m}_{p}_{t}"
                    )
        
        for v in range(self.num_vehicles):
            model.addCons(
                quicksum(load_vars[v, n] for n in range(self.num_nodes)) <= vehicle_capacities[v],
                name=f"VehicleCapacity_{v}"
            )
        
        for v in range(self.num_vehicles):
            model.addCons(
                quicksum(distance_traveled_vars[v, i, j] * vehicle_distances[v][i][j] for i in range(self.num_nodes) for j in range(self.num_nodes)) <= self.max_distance_per_vehicle,
                name=f"VehicleDistance_{v}"
            )
        
        model.optimize()
        
        return model.getStatus(), model.getObjVal()
    
if __name__ == '__main__':
    seed = 42
    parameters = {
        'num_machines': 10,
        'num_products': 5,
        'num_periods': 140,
        'num_vehicles': 8,
        'num_nodes': 20,
        'max_distance_per_vehicle': 500,
        'product_prices': (100, 150, 200, 250, 300),
    }
    
    factory_production = FactoryProduction(parameters, seed=seed)
    instance = factory_production.get_instance()
    solve_status, solve_val = factory_production.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Objective Value: {solve_val:.2f}")