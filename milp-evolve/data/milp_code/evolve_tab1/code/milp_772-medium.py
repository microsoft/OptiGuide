import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class InfrastructurePlanning:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def gamma_int(self, size, shape, scale, lower_bound, upper_bound):
        return np.clip(
            np.round(np.random.gamma(shape, scale, size)), 
            lower_bound, 
            upper_bound
        ).astype(int)

    def quadratic_costs(self):
        scaling = 5.0
        rand = lambda n, m: np.random.rand(n, m)
        costs = scaling * (rand(self.number_of_zones, 1) - rand(1, self.number_of_facilities))**2
        return costs

    def generate_instance(self):
        if self.dynamic_range:
            range_factor = np.random.uniform(0.5, 2.0)
            min_range = int(self.base_range * range_factor)
            max_range = int(self.base_range * range_factor * 2)
        else:
            min_range = self.min_range
            max_range = self.max_range

        costs = np.random.randint(min_range, max_range, self.number_of_zones)

        if self.scheme == 'uncorrelated':
            new_costs = np.random.randint(min_range, max_range, self.number_of_zones)
        elif self.scheme == 'weakly correlated':
            new_costs = np.apply_along_axis(
                lambda x: np.random.randint(x[0], x[1]),
                axis=0,
                arr=np.vstack([
                    np.maximum(costs - (max_range-min_range), 1),
                               costs + (max_range-min_range)]))
        else:
            raise NotImplementedError

        capacities = np.zeros(self.number_of_facilities, dtype=int)
        capacities[:-1] = np.random.randint(0.4 * costs.sum() // self.number_of_facilities,
                                            0.6 * costs.sum() // self.number_of_facilities,
                                            self.number_of_facilities - 1)
        capacities[-1] = 0.5 * costs.sum() - capacities[:-1].sum()

        demands = self.gamma_int(
            self.number_of_zones, 
            self.demand_shape, 
            self.demand_scale, 
            self.demand_lower, 
            self.demand_upper
        )
                                                
        new_infra_costs = (
            self.gamma_int(
                self.number_of_facilities, 
                self.infra_cost_shape, 
                self.infra_cost_scale, 
                self.infra_cost_lower, 
                self.infra_cost_upper
            ) * np.sqrt(capacities)
        )

        highways_costs = self.quadratic_costs() * demands[:, np.newaxis]

        capacities = capacities * self.Ration * np.sum(demands) / np.sum(capacities)

        res = {'costs': costs,
               'new_costs': new_costs,
               'capacities': capacities,
               'demands': demands,
               'new_infra_costs': new_infra_costs,
               'highways_costs': highways_costs}

        rand = lambda n, m: np.random.randint(1, 10, size=(n, m))
        additional_infra_costs = rand(self.number_of_zones, self.number_of_facilities)
        res['additional_infra_costs'] = additional_infra_costs

        return res
        

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        costs = instance['costs']
        new_costs = instance['new_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        new_infra_costs = instance['new_infra_costs']
        highways_costs = instance['highways_costs']
        additional_infra_costs = instance['additional_infra_costs']
        
        number_of_zones = len(costs)
        number_of_facilities = len(capacities)
        
        model = Model("InfrastructurePlanning")
        var_names = {}

        # Decision variables: cost[i][j] = cost of facility i in zone j
        for i in range(number_of_zones):
            for j in range(number_of_facilities):
                var_names[(i, j)] = model.addVar(vtype="C", name=f"cost_{i}_{j}")

        maintenance_schedule = {j: model.addVar(vtype="I", name=f"maintenance_{j}") for j in range(number_of_facilities)}
        demand_serve = {(i, j): model.addVar(vtype="C", name=f"serve_{i}_{j}") for i in range(number_of_zones) for j in range(number_of_facilities)}

        # Objective: Minimize total new infrastructure costs plus maintenance and highways costs
        objective_expr = quicksum(costs[i] * var_names[(i, j)] for i in range(number_of_zones) for j in range(number_of_facilities)) + \
                         quicksum(new_infra_costs[j] * maintenance_schedule[j] for j in range(number_of_facilities)) + \
                         quicksum(highways_costs[i, j] * demand_serve[i, j] for i in range(number_of_zones) for j in range(number_of_facilities)) + \
                         quicksum(additional_infra_costs[i, j] * var_names[(i, j)] for i in range(number_of_zones) for j in range(number_of_facilities))

        # Constraints: Each zone must get its demand met
        for i in range(number_of_zones):
            model.addCons(
                quicksum(var_names[(i, j)] for j in range(number_of_facilities)) >= 1,
                f"DemandAssignment_{i}"
            )

        # Constraints: Maintenance schedule limits at each facility
        for j in range(number_of_facilities):
            model.addCons(
                quicksum(costs[i] * var_names[(i, j)] for i in range(number_of_zones)) <= capacities[j],
                f"FacilityCapacity_{j}"
            )
            
        # Constraints: Each zone must be served
        for i in range(number_of_zones):
            model.addCons(
                quicksum(demand_serve[i, j] for j in range(number_of_facilities)) >= demands[i],
                f"Serve_Demands_{i}"
            )
        
        # Constraints: Capacity limits at each facility
        for j in range(number_of_facilities):
            model.addCons(
                quicksum(demand_serve[i, j] for i in range(number_of_zones)) <= capacities[j] * maintenance_schedule[j],
                f"Facility_Capacity_{j}"
            )

        model.setObjective(objective_expr, "minimize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_zones': 112,
        'number_of_facilities': 270,
        'min_range': 540,
        'max_range': 2250,
        'base_range': 900,
        'dynamic_range': 0,
        'scheme': 'weakly correlated',
        'demand_shape': 75.0,
        'demand_scale': 6.0,
        'demand_lower': 400,
        'demand_upper': 1000,
        'infra_cost_shape': 4.0,
        'infra_cost_scale': 21.0,
        'infra_cost_lower': 2025,
        'infra_cost_upper': 2700,
        'Ration': 2400.0,
        'additional_infra_cost_mean': 75,
        'additional_infra_cost_std': 9,
        'additional_infra_cost_lower': 10,
        'additional_infra_cost_upper': 6,
    }

    infrastructure_planning = InfrastructurePlanning(parameters, seed=seed)
    instance = infrastructure_planning.generate_instance()
    solve_status, solve_time = infrastructure_planning.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")