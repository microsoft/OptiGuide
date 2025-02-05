import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum, multidict

class HospitalPatientTransferOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        assert self.n_hospitals > 0 and self.n_zones > 0
        assert self.min_hospital_cost >= 0 and self.max_hospital_cost >= self.min_hospital_cost
        assert self.min_transport_cost >= 0 and self.max_transport_cost >= self.min_transport_cost
        assert self.min_bed_capacity > 0 and self.max_bed_capacity >= self.min_bed_capacity

        hospital_costs = np.random.randint(self.min_hospital_cost, self.max_hospital_cost + 1, self.n_hospitals)
        transport_costs = np.random.randint(self.min_transport_cost, self.max_transport_cost + 1, (self.n_hospitals, self.n_zones))
        bed_capacities = np.random.randint(self.min_bed_capacity, self.max_bed_capacity + 1, self.n_hospitals)
        patient_needs = np.random.randint(1, 10, self.n_zones)
        equipment_limits = np.random.uniform(self.min_equipment_limit, self.max_equipment_limit, self.n_hospitals)
        distances = np.random.uniform(0, self.max_distance, (self.n_hospitals, self.n_zones))

        # Create a flow network graph using NetworkX
        G = nx.DiGraph()
        node_pairs = []
        for h in range(self.n_hospitals):
            for z in range(self.n_zones):
                G.add_edge(f"hospital_{h}", f"zone_{z}")
                node_pairs.append((f"hospital_{h}", f"zone_{z}"))
                
        return {
            "hospital_costs": hospital_costs,
            "transport_costs": transport_costs,
            "bed_capacities": bed_capacities,
            "patient_needs": patient_needs,
            "equipment_limits": equipment_limits,
            "distances": distances,
            "graph": G,
            "node_pairs": node_pairs
        }

    def solve(self, instance):
        hospital_costs = instance['hospital_costs']
        transport_costs = instance['transport_costs']
        bed_capacities = instance['bed_capacities']
        patient_needs = instance['patient_needs']
        equipment_limits = instance['equipment_limits']
        distances = instance['distances']
        G = instance['graph']
        node_pairs = instance['node_pairs']

        model = Model("HospitalPatientTransferOptimization")
        n_hospitals = len(hospital_costs)
        n_zones = len(transport_costs[0])
        
        # Decision variables
        open_vars = {h: model.addVar(vtype="B", name=f"Hospital_{h}") for h in range(n_hospitals)}
        flow_vars = {(u, v): model.addVar(vtype="C", name=f"Flow_{u}_{v}") for u, v in node_pairs}

        # Objective: minimize the total cost including hospital operating costs and transport costs.
        model.setObjective(
            quicksum(hospital_costs[h] * open_vars[h] for h in range(n_hospitals)) +
            quicksum(transport_costs[h, int(v.split('_')[1])] * flow_vars[(u, v)] for (u, v) in node_pairs for h in range(n_hospitals) if u == f'hospital_{h}'),
            "minimize"
        )

        # Flow conservation constraint for each zone
        for z in range(n_zones):
            model.addCons(
                quicksum(flow_vars[(u, f"zone_{z}")] for u in G.predecessors(f"zone_{z}")) == patient_needs[z], 
                f"Zone_{z}_Flow_Conservation"
            )

        # Constraints: Hospitals only send flows if they are open
        for h in range(n_hospitals):
            for z in range(n_zones):
                model.addCons(
                    flow_vars[(f"hospital_{h}", f"zone_{z}")] <= equipment_limits[h] * open_vars[h], 
                    f"Hospital_{h}_Flow_Limit_{z}"
                )

        # Constraints: Hospitals cannot exceed their bed capacities
        for h in range(n_hospitals):
            model.addCons(
                quicksum(flow_vars[(f"hospital_{h}", f"zone_{z}")] for z in range(n_zones)) <= bed_capacities[h], 
                f"Hospital_{h}_Total_Flow_Capacity"
            )

        # Movement distance constraint (Critical zone)
        for z in range(n_zones):
            model.addCons(
                quicksum(open_vars[h] for h in range(n_hospitals) if distances[h, z] <= self.max_distance) >= 1, 
                f"Zone_{z}_Critical"
            )

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_hospitals': 70,
        'n_zones': 270,
        'min_transport_cost': 250,
        'max_transport_cost': 1200,
        'min_hospital_cost': 800,
        'max_hospital_cost': 1875,
        'min_bed_capacity': 1875,
        'max_bed_capacity': 2000,
        'min_equipment_limit': 360,
        'max_equipment_limit': 2136,
        'max_distance': 1050,
    }
    
    hospital_optimizer = HospitalPatientTransferOptimization(parameters, seed=seed)
    instance = hospital_optimizer.generate_instance()
    solve_status, solve_time, objective_value = hospital_optimizer.solve(instance)
    
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")