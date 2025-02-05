import random
import time
import numpy as np
from pyscipopt import Model, quicksum

class WarehouseLayoutOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# data generation #################
    def generate_instance(self):
        assert self.n_facilities > 0 and self.n_units > 0
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_facility_cost >= 0 and self.max_facility_cost >= self.min_facility_cost
        assert self.min_facility_space > 0 and self.max_facility_space >= self.min_facility_space
        
        facility_costs = np.random.randint(self.min_facility_cost, self.max_facility_cost + 1, self.n_facilities)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_facilities, self.n_units))
        spaces = np.random.randint(self.min_facility_space, self.max_facility_space + 1, self.n_facilities)
        demands = np.random.randint(1, 10, self.n_units)
        
        road_distances = np.random.rand(self.n_facilities) * self.max_distance_to_road  # random distances to roads
        facility_distances = np.random.rand(self.n_facilities, self.n_facilities) * 2 * self.max_facility_distance  # adjust distance range
        
        return {
            "facility_costs": facility_costs,
            "transport_costs": transport_costs,
            "spaces": spaces,
            "demands": demands,
            "road_distances": road_distances,
            "facility_distances": facility_distances
        }
        
    ################# PySCIPOpt modeling #################
    def solve(self, instance):
        facility_costs = instance['facility_costs']
        transport_costs = instance['transport_costs']
        spaces = instance['spaces']
        demands = instance['demands']
        road_distances = instance['road_distances']
        facility_distances = instance['facility_distances']
        
        model = Model("WarehouseLayoutOptimization")
        n_facilities = len(facility_costs)
        n_units = len(transport_costs[0])
        
        # Decision variables
        facility_vars = {f: model.addVar(vtype="B", name=f"Facility_{f}") for f in range(n_facilities)}
        transport_vars = {(f, u): model.addVar(vtype="B", name=f"Facility_{f}_Unit_{u}") for f in range(n_facilities) for u in range(n_units)}
        road_distance_vars = {f: model.addVar(vtype="C", name=f"RoadDist_{f}", lb=0, ub=self.max_distance_to_road) for f in range(n_facilities)}
        facility_distance_vars = {(i, j): model.addVar(vtype="C", name=f"Dist_{i}_{j}", lb=0, ub=self.max_facility_distance) for i in range(n_facilities) for j in range(n_facilities)}

        # Objective: minimize total cost with penalties for road distance & facility distance
        model.setObjective(
            quicksum(facility_costs[f] * facility_vars[f] for f in range(n_facilities)) +
            quicksum(transport_costs[f, u] * transport_vars[f, u] for f in range(n_facilities) for u in range(n_units)) +
            quicksum(self.road_distance_penalty * (road_distances[f] - road_distance_vars[f]) for f in range(n_facilities)) +
            quicksum(self.facility_distance_penalty * (facility_distances[i, j] - facility_distance_vars[(i, j)]) for i in range(n_facilities) for j in range(n_facilities) if i != j),
            "minimize"
        )
        
        # Constraints: Each unit demand is met by exactly one facility
        for u in range(n_units):
            model.addCons(quicksum(transport_vars[f, u] for f in range(n_facilities)) == 1, f"Unit_{u}_Demand")
        
        # Constraints: Only open facilities can serve units
        for f in range(n_facilities):
            for u in range(n_units):
                model.addCons(transport_vars[f, u] <= facility_vars[f], f"Facility_{f}_Serve_{u}")
        
        # Constraints: Facilities cannot exceed their space
        for f in range(n_facilities):
            model.addCons(quicksum(demands[u] * transport_vars[f, u] for u in range(n_units)) <= spaces[f], f"Facility_{f}_Space")
        
        # Constraints: Each facility's distance to the nearest major road
        for f in range(n_facilities):
            model.addCons(road_distance_vars[f] <= road_distances[f], f"Road_Dist_{f}")
        
        # Constraints: Distance between any two facilities should not exceed the specified maximum
        for i in range(n_facilities):
            for j in range(n_facilities):
                if i != j:
                    model.addCons(facility_distance_vars[(i, j)] <= self.max_facility_distance, f"Facility_Dist_{i}_{j}")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_facilities': 25,
        'n_units': 252,
        'min_transport_cost': 0,
        'max_transport_cost': 1125,
        'min_facility_cost': 3000,
        'max_facility_cost': 5000,
        'min_facility_space': 787,
        'max_facility_space': 900,
        'max_distance_to_road': 67.5,
        'max_facility_distance': 15.0,
        'road_distance_penalty': 75,
        'facility_distance_penalty': 0,
    }
    warehouse_layout_optimizer = WarehouseLayoutOptimization(parameters, seed=42)
    instance = warehouse_layout_optimizer.generate_instance()
    solve_status, solve_time, objective_value = warehouse_layout_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")