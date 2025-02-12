import random
import time
import numpy as np
from pyscipopt import Model, quicksum
import networkx as nx

class DeliveryOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_instance(self):
        zones = range(self.number_of_zones)
        warehouses = range(self.potential_warehouses)

        # Generating zone parcel demand
        parcel_demand = np.random.gamma(shape=self.demand_shape, scale=self.demand_scale, size=self.number_of_zones).astype(int)
        parcel_demand = np.clip(parcel_demand, self.min_demand, self.max_demand)

        # Generating the delivery times
        delivery_times = np.random.beta(self.alpha_delivery_time, self.beta_delivery_time, size=self.number_of_zones)

        # Distance matrix between zones and potential warehouse locations
        G = nx.barabasi_albert_graph(self.number_of_zones, self.number_of_graph_edges)
        distances = nx.floyd_warshall_numpy(G) * np.random.uniform(self.min_distance_multiplier, self.max_distance_multiplier)

        # Generating warehouse costs and capacities
        warehouse_costs = np.random.gamma(self.warehouse_cost_shape, self.warehouse_cost_scale, size=self.potential_warehouses)
        warehouse_capacities = np.random.gamma(self.warehouse_capacity_shape, self.warehouse_capacity_scale, size=self.potential_warehouses).astype(int)

        # Sustainable energy availability
        energy_availability = np.random.weibull(self.energy_shape, size=self.potential_warehouses) * self.energy_scale

        # Breakpoints for piecewise linear functions
        d_breakpoints = np.linspace(0, 1, self.num_segments + 1)
        transport_costs_segments = np.random.uniform(self.min_transport_cost, self.max_transport_cost, self.num_segments)

        # Budget for each infrastructure upgrade
        operational_budgets = np.random.uniform(self.min_operational_budget, self.max_operational_budget, size=self.potential_warehouses)

        # Emission rates
        emission_rates = np.random.uniform(self.min_emission_rate, self.max_emission_rate, size=self.number_of_vehicles)

        # Eco-friendly vehicle flag
        eco_friendly_flag = np.random.binomial(1, self.eco_friendly_prob, size=self.number_of_vehicles)

        # Generating new budget and resource limitations
        total_budget = np.random.normal(self.avg_budget, self.budget_stddev)
        total_resources = np.random.gamma(self.resource_shape, self.resource_scale, size=self.resource_types)

        res = {
            'parcel_demand': parcel_demand,
            'delivery_times': delivery_times,
            'distances': distances,
            'warehouse_costs': warehouse_costs,
            'warehouse_capacities': warehouse_capacities,
            'energy_availability': energy_availability,
            'd_breakpoints': d_breakpoints,
            'transport_costs_segments': transport_costs_segments,
            'operational_budgets': operational_budgets,
            'emission_rates': emission_rates,
            'eco_friendly_flag': eco_friendly_flag,
            'total_budget': total_budget,
            'total_resources': total_resources
        }

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        parcel_demand = instance['parcel_demand']
        delivery_times = instance['delivery_times']
        distances = instance['distances']
        warehouse_costs = instance['warehouse_costs']
        warehouse_capacities = instance['warehouse_capacities']
        energy_availability = instance['energy_availability']
        d_breakpoints = instance['d_breakpoints']
        transport_costs_segments = instance['transport_costs_segments']
        operational_budgets = instance['operational_budgets']
        emission_rates = instance['emission_rates']
        eco_friendly_flag = instance['eco_friendly_flag']
        total_budget = instance['total_budget']
        total_resources = instance['total_resources']

        zones = range(len(parcel_demand))
        warehouses = range(len(warehouse_costs))
        segments = range(len(transport_costs_segments))
        vehicles = range(len(emission_rates))
        resource_types = range(len(total_resources))

        model = Model("DeliveryOptimization")
        x = {}  # Binary variable: 1 if warehouse is placed at location l
        y = {}  # Binary variable: 1 if a delivery route from z to w exists
        f = {}  # Integer variable: Delivery frequency
        d = {}  # Continuous variable: Fraction of demand satisfied
        z_zl_s = {}  # Binary variable for piecewise linear segment
        c = {}  # Continuous variable: Cost of operational upgrade at location l
        emissions_zl = {}  # Emissions for delivery route z to l
        eco_friendly_zl = {}  # Binary variable indicating if an eco-friendly vehicle is used
        b = model.addVar(vtype="B", name="budget_exceeded")  # Binary variable: 1 if budget is exceeded
        r_usage = {}  # Resource usage
        r_exceed = {}  # Resource exceed binary variable

        # Decision variables
        for l in warehouses:
            x[l] = model.addVar(vtype="B", name=f"x_{l}")
            c[l] = model.addVar(vtype="C", name=f"c_{l}", lb=0)
            for z in zones:
                y[z, l] = model.addVar(vtype="B", name=f"y_{z}_{l}")
                f[z, l] = model.addVar(vtype="I", name=f"f_{z}_{l}", lb=0, ub=self.max_delivery_frequency)
                d[z, l] = model.addVar(vtype="C", name=f"d_{z}_{l}", lb=0, ub=1)
                emissions_zl[z, l] = model.addVar(vtype="C", name=f"emissions_{z}_{l}", lb=0)
                eco_friendly_zl[z, l] = model.addVar(vtype="B", name=f"eco_friendly_{z}_{l}")

                for s in segments:
                    z_zl_s[z, l, s] = model.addVar(vtype="B", name=f"z_{z}_{l}_{s}")

        # Resource usage and exceed binary variables
        for r in resource_types:
            r_usage[r] = model.addVar(vtype="C", name=f"r_usage_{r}", lb=0)
            r_exceed[r] = model.addVar(vtype="B", name=f"r_exceed_{r}")

        # Objective: Minimize delivery cost + warehouse costs + delivery impact + maximize delivery satisfaction + operational costs + minimize emissions + penalty for budget exceedance
        obj_expr = quicksum(distances[z][l] * y[z, l] * parcel_demand[z] * delivery_times[z] for z in zones for l in warehouses) + \
                   quicksum(warehouse_costs[l] * x[l] for l in warehouses) + \
                   quicksum((1 - d[z, l]) * parcel_demand[z] for z in zones for l in warehouses) - \
                   quicksum(d[z, l] for z in zones for l in warehouses) + \
                   quicksum(transport_costs_segments[s] * z_zl_s[z, l, s] for z in zones for l in warehouses for s in segments) + \
                   quicksum(c[l] for l in warehouses) + \
                   quicksum(emissions_zl[z, l] for z in zones for l in warehouses) + \
                   10000 * b  # Penalty for budget exceedance

        model.setObjective(obj_expr, "minimize")

        # Constraints: Each zone must have access to one delivery route
        for z in zones:
            model.addCons(
                quicksum(y[z, l] for l in warehouses) == 1,
                f"Access_{z}"
            )

        # Constraints: Warehouse capacity must not be exceeded
        for l in warehouses:
            model.addCons(
                quicksum(y[z, l] * parcel_demand[z] * delivery_times[z] * d[z, l] for z in zones) <= warehouse_capacities[l] * x[l],
                f"Capacity_{l}"
            )

        # Constraints: Sustainable energy availability at each warehouse
        for l in warehouses:
            model.addCons(
                quicksum(y[z, l] * delivery_times[z] * f[z, l] for z in zones) <= energy_availability[l],
                f"Energy_{l}"
            )

        # Constraints: Delivery frequency related to demand satisfaction
        for z in zones:
            for l in warehouses:
                model.addCons(
                    f[z, l] == parcel_demand[z] * d[z, l],
                    f"Frequency_Demand_{z}_{l}"
                )

        # Piecewise Linear Constraints: Ensure each demand fraction lies within a segment
        for z in zones:
            for l in warehouses:
                model.addCons(
                    quicksum(z_zl_s[z, l, s] for s in segments) == 1,
                    f"Segment_{z}_{l}"
                )

                for s in segments[:-1]:
                    model.addCons(
                        d[z, l] >= d_breakpoints[s] * z_zl_s[z, l, s],
                        f"LowerBoundSeg_{z}_{l}_{s}"
                    )
                    model.addCons(
                        d[z, l] <= d_breakpoints[s + 1] * z_zl_s[z, l, s] + (1 - z_zl_s[z, l, s]) * d_breakpoints[-1],
                        f"UpperBoundSeg_{z}_{l}_{s}"
                    )

        # Constraints: Operational upgrade budget limit
        for l in warehouses:
            model.addCons(
                c[l] <= operational_budgets[l] * x[l],
                f"Budget_{l}"
            )

        # Constraints: Emissions limit for each route
        for z in zones:
            for l in warehouses:
                model.addCons(
                    emissions_zl[z, l] <= self.emission_limit * y[z, l],
                    f"EmissionLimit_{z}_{l}"
                )

        # Constraints: Ensure the use of eco-friendly vehicles on a minimum percentage of routes
        model.addCons(
            quicksum(eco_friendly_zl[z, l] * y[z, l] for z in zones for l in warehouses) >=
            self.min_eco_friendly_routes * quicksum(y[z, l] for z in zones for l in warehouses),
            "MinEcoFriendlyRoutes"
        )

        # New Constraints: Ensure total budget limit for all costs
        total_costs_expr = quicksum(warehouse_costs[l] * x[l] for l in warehouses) + \
                           quicksum(transport_costs_segments[s] * z_zl_s[z, l, s] for z in zones for l in warehouses for s in segments) + \
                           quicksum(c[l] for l in warehouses) + \
                           quicksum(distances[z][l] * y[z, l] * parcel_demand[z] * delivery_times[z] for z in zones for l in warehouses)

        model.addCons(
            total_costs_expr <= total_budget + b * total_budget,
            "TotalBudgetLimit"
        )

        # New Constraints: Ensure total resource usage does not exceed available resources
        for r in resource_types:
            total_resource_usage = quicksum(y[z, l] * f[z, l] * (r + 1) for z in zones for l in warehouses)
            model.addCons(
                total_resource_usage <= total_resources[r] + r_exceed[r] * total_resources[r],
                f"ResourceLimit_{r}"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'number_of_zones': 56,
        'potential_warehouses': 50,
        'demand_shape': 12,
        'demand_scale': 600,
        'min_demand': 2500,
        'max_demand': 3000,
        'alpha_delivery_time': 54.0,
        'beta_delivery_time': 11.25,
        'number_of_graph_edges': 15,
        'min_distance_multiplier': 7,
        'max_distance_multiplier': 36,
        'warehouse_cost_shape': 5.0,
        'warehouse_cost_scale': 2000.0,
        'warehouse_capacity_shape': 37.5,
        'warehouse_capacity_scale': 2000.0,
        'energy_shape': 0.66,
        'energy_scale': 10000,
        'max_delivery_frequency': 25,
        'num_segments': 50,
        'min_transport_cost': 7.0,
        'max_transport_cost': 7.5,
        'min_operational_budget': 1000.0,
        'max_operational_budget': 10000.0,
        'emission_limit': 100.0,
        'min_eco_friendly_routes': 0.59,
        'number_of_vehicles': 200,
        'min_emission_rate': 80.0,
        'max_emission_rate': 1050.0,
        'eco_friendly_prob': 0.17,
        'avg_budget': 500000.0,
        'budget_stddev': 50000.0,
        'resource_shape': 2.0,
        'resource_scale': 1500.0,
        'resource_types': 2,
    }

    delivery_optimization = DeliveryOptimization(parameters, seed=seed)
    instance = delivery_optimization.generate_instance()
    solve_status, solve_time = delivery_optimization.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")