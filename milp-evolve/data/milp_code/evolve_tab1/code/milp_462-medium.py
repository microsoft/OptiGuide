import random
import time
import numpy as np
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

class SupplyChainNetworkOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_facility_locations(self):
        n_nodes = np.random.randint(self.min_facilities, self.max_facilities)
        facilities = np.random.choice(range(1000), size=n_nodes, replace=False)   # Random unique facility IDs
        return facilities

    def generate_demand_nodes(self, n_facilities):
        n_demand_nodes = np.random.randint(self.min_demand_nodes, self.max_demand_nodes)
        demand_nodes = np.random.choice(range(1000, 2000), size=n_demand_nodes, replace=False)
        demands = {node: np.random.randint(self.min_demand, self.max_demand) for node in demand_nodes}
        return demand_nodes, demands

    def generate_transportation_costs(self, facilities, demand_nodes):
        costs = {(f, d): np.random.uniform(self.min_transport_cost, self.max_transport_cost) for f in facilities for d in demand_nodes}
        return costs

    def generate_facility_capacities(self, facilities):
        capacities = {f: np.random.randint(self.min_capacity, self.max_capacity) for f in facilities}
        return capacities

    def get_instance(self):
        facilities = self.generate_facility_locations()
        demand_nodes, demands = self.generate_demand_nodes(len(facilities))
        transport_costs = self.generate_transportation_costs(facilities, demand_nodes)
        capacities = self.generate_facility_capacities(facilities)
        
        facility_costs = {f: np.random.uniform(self.min_facility_cost, self.max_facility_cost) for f in facilities}
        processing_costs = {f: np.random.uniform(self.min_processing_cost, self.max_processing_cost) for f in facilities}
        auxiliary_costs = {f: np.random.uniform(self.min_auxiliary_cost, self.max_auxiliary_cost) for f in facilities}
        
        return {
            'facilities': facilities,
            'demand_nodes': demand_nodes,
            'transport_costs': transport_costs,
            'capacities': capacities,
            'demands': demands,
            'facility_costs': facility_costs,
            'processing_costs': processing_costs,
            'auxiliary_costs': auxiliary_costs,
        }

    def solve(self, instance):
        facilities = instance['facilities']
        demand_nodes = instance['demand_nodes']
        transport_costs = instance['transport_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        facility_costs = instance['facility_costs']
        processing_costs = instance['processing_costs']
        auxiliary_costs = instance['auxiliary_costs']

        model = Model("SupplyChainNetworkOptimization")

        NetworkNode_location_vars = {f: model.addVar(vtype="B", name=f"NN_Loc_{f}") for f in facilities}
        Allocation_vars = {(f, d): model.addVar(vtype="C", name=f"Alloc_{f}_{d}") for f in facilities for d in demand_nodes}
        Transportation_vars = {(f, d): model.addVar(vtype="C", name=f"Trans_{f}_{d}") for f in facilities for d in demand_nodes}
        Assembly_vars = {f: model.addVar(vtype="C", name=f"Assembly_{f}") for f in facilities}
        Ancillary_vars = {f: model.addVar(vtype="C", name=f"Ancillary_{f}") for f in facilities}

        # Objective function
        total_cost = quicksum(NetworkNode_location_vars[f] * facility_costs[f] for f in facilities)
        total_cost += quicksum(Allocation_vars[f, d] * transport_costs[f, d] for f in facilities for d in demand_nodes)
        total_cost += quicksum(Assembly_vars[f] * processing_costs[f] for f in facilities)
        total_cost += quicksum(Ancillary_vars[f] * auxiliary_costs[f] for f in facilities)
        
        model.setObjective(total_cost, "minimize")

        # Constraints
        # Convex Hull Formulation:
        for f in facilities:
            model.addCons(
                quicksum(Allocation_vars[f, d] for d in demand_nodes) <= capacities[f] * NetworkNode_location_vars[f],
                name=f"Capacity_{f}"
            )

        for d in demand_nodes:
            model.addCons(
                quicksum(Allocation_vars[f, d] for f in facilities) >= demands[d],
                name=f"DemandSatisfaction_{d}"
            )

        for f in facilities:
            model.addCons(
                Assembly_vars[f] <= capacities[f],
                name=f"AssemblyLimit_{f}"
            )

        for f in facilities:
            model.addCons(
                Ancillary_vars[f] <= capacities[f] * 0.15,  # Assume ancillary operations are limited to 15% of the capacity
                name=f"AncillaryLimit_{f}"
            )

        model.setParam('limits/time', 10 * 60)  # Set a time limit of 10 minutes for solving
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_facilities': 200,
        'max_facilities': 1000,
        'min_demand_nodes': 20,
        'max_demand_nodes': 100,
        'min_demand': 40,
        'max_demand': 700,
        'min_transport_cost': 1.0,
        'max_transport_cost': 50.0,
        'min_capacity': 400,
        'max_capacity': 500,
        'min_facility_cost': 50000,
        'max_facility_cost': 200000,
        'min_processing_cost': 2000,
        'max_processing_cost': 2000,
        'min_auxiliary_cost': 800,
        'max_auxiliary_cost': 800,
    }
    
    optimizer = SupplyChainNetworkOptimization(parameters, seed=seed)
    instance = optimizer.get_instance()
    status, solve_time = optimizer.solve(instance)
    print(f"Solve Status: {status}")
    print(f"Solve Time: {solve_time:.2f} seconds")