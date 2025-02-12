import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class SmartCityTransportPlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    ################# data generation #################
    def randint(self, size, interval):
        return np.random.randint(interval[0], interval[1], size)
    
    def normal_dist(self, size, mean, stddev):
        return np.random.normal(mean, stddev, size)

    def unit_transportation_costs(self):
        scaling = 10.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * np.sqrt(
            (rand(self.n_neighborhoods, 1) - rand(1, self.n_factories))**2 +
            (rand(self.n_neighborhoods, 1) - rand(1, self.n_factories))**2
        )
        return costs

    def gamma_dist(self, size, shape, scale):
        return np.random.gamma(shape, scale, size)
    
    def generate_instance(self):
        transportation_demands = self.normal_dist(self.n_neighborhoods, self.demand_mean, self.demand_stddev).astype(int)
        depot_capacities = self.gamma_dist(self.n_factories, self.capacity_shape, self.capacity_scale).astype(int)
        tram_station_costs = (
            self.normal_dist(self.n_factories, self.setup_cost_mean, self.setup_cost_stddev).astype(int) +
            self.randint(self.n_factories, self.setup_cost_interval)
        )
        transportation_costs = self.unit_transportation_costs() * transportation_demands[:, np.newaxis]
        tram_cost_budgets = self.normal_dist(self.n_factories, self.tram_cost_mean, self.tram_cost_stddev).astype(int)

        depot_capacities = depot_capacities * self.ratio * np.sum(transportation_demands) / np.sum(depot_capacities)
        depot_capacities = np.round(depot_capacities)
        
        res = {
            'transportation_demands': transportation_demands,
            'depot_capacities': depot_capacities,
            'tram_station_costs': tram_station_costs,
            'transportation_costs': transportation_costs,
            'tram_cost_budgets': tram_cost_budgets
        }
        
        return res

    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        transportation_demands = instance['transportation_demands']
        depot_capacities = instance['depot_capacities']
        tram_station_costs = instance['tram_station_costs']
        transportation_costs = instance['transportation_costs']
        tram_cost_budgets = instance['tram_cost_budgets']
        
        n_neighborhoods = len(transportation_demands)
        n_factories = len(depot_capacities)
        
        model = Model("SmartCityTransportPlanning")
        
        # Decision variables
        factory_depot = {j: model.addVar(vtype="B", name=f"FactoryDepot_{j}") for j in range(n_factories)}
        tram_station = {j: model.addVar(vtype="B", name=f"TramStation_{j}") for j in range(n_factories)}
        if self.continuous_distribution:
            route = {(i, j): model.addVar(vtype="C", name=f"Route_{i}_{j}") for i in range(n_neighborhoods) for j in range(n_factories)}
        else:
            route = {(i, j): model.addVar(vtype="B", name=f"Route_{i}_{j}") for i in range(n_neighborhoods) for j in range(n_factories)}
        tram_service = {j: model.addVar(vtype="C", name=f"TramService_{j}") for j in range(n_factories)}

        # Objective: minimize the total cost
        objective_expr = quicksum(tram_station_costs[j] * factory_depot[j] for j in range(n_factories)) + \
                         quicksum(transportation_costs[i, j] * route[i, j] for i in range(n_neighborhoods) for j in range(n_factories)) + \
                         quicksum(tram_cost_budgets[j] * tram_service[j] for j in range(n_factories))
        
        model.setObjective(objective_expr, "minimize")
        
        # Constraints: transportation demand must be met
        for i in range(n_neighborhoods):
            model.addCons(quicksum(route[i, j] for j in range(n_factories)) >= 1, f"TransportationDemand_{i}")
        
        # Constraints: depot capacity limits
        for j in range(n_factories):
            model.addCons(quicksum(route[i, j] * transportation_demands[i] for i in range(n_neighborhoods)) <= depot_capacities[j] * factory_depot[j], f"DepotCapacity_{j}")

        # Constraints: tram service budget limits
        for j in range(n_factories):
            model.addCons(tram_service[j] <= self.max_tram_cost_budget, f"TramCostBudget_{j}")
        
        # Constraints: annual transport plan
        total_transport = np.sum(transportation_demands)
        model.addCons(quicksum(depot_capacities[j] * factory_depot[j] for j in range(n_factories)) >= self.annual_transport_plan * total_transport, "AnnualTransportPlan")
        
        # Symmetry-breaking constraints using lexicographical order
        for j in range(n_factories - 1):
            model.addCons(factory_depot[j] >= factory_depot[j + 1], f"Symmetry_{j}_{j+1}")

        # Tram Station Constraints
        for j in range(n_factories):
            model.addCons(tram_station[j] == 1 - factory_depot[j], f"TramStation_{j}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_neighborhoods': 50,
        'n_factories': 350,
        'demand_mean': 150,
        'demand_stddev': 40,
        'capacity_shape': 100.0,
        'capacity_scale': 1875.0,
        'setup_cost_mean': 1100.0,
        'setup_cost_stddev': 300.0,
        'setup_cost_interval': (0, 10),
        'tram_cost_mean': 6000,
        'tram_cost_stddev': 900,
        'max_tram_cost_budget': 7500,
        'annual_transport_plan': 0.75,
        'ratio': 10.50,
        'continuous_distribution': 0,
    }

    transport_planner = SmartCityTransportPlanning(parameters, seed=seed)
    instance = transport_planner.generate_instance()
    solve_status, solve_time = transport_planner.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")