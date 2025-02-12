import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HospitalResourceAllocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.n_departments > 0 and self.n_patient_types >= self.n_departments
        assert self.min_treatment_cost >= 0 and self.max_treatment_cost >= self.min_treatment_cost
        assert self.min_equipment_cost >= 0 and self.max_equipment_cost >= self.min_equipment_cost
        assert self.min_staff_cost > 0 and self.max_staff_cost >= self.min_staff_cost

        treatment_costs = np.random.randint(self.min_treatment_cost, self.max_treatment_cost + 1, self.n_departments)
        equipment_costs = np.random.randint(self.min_equipment_cost, self.max_equipment_cost + 1, (self.n_departments, self.n_patient_types))
        staff_costs = np.random.randint(self.min_staff_cost, self.max_staff_cost + 1, self.n_departments)
        qalys_gains = np.random.uniform(10, 100, self.n_patient_types)
        equipment_maintenance = np.random.uniform(0.5, 2.0, self.n_departments).tolist()
        staff_availability = np.random.uniform(50, 200, self.n_patient_types).tolist()
        treatment_fluctuation = np.random.normal(1, 0.2, self.n_patient_types).tolist()
        ordered_patient_types = list(np.random.permutation(self.n_patient_types))

        # Generate complex interaction matrix representing a random graph
        G = nx.barabasi_albert_graph(self.n_patient_types, 5)
        adjacency_matrix = nx.adjacency_matrix(G).toarray()

        return {
            "treatment_costs": treatment_costs,
            "equipment_costs": equipment_costs,
            "staff_costs": staff_costs,
            "qalys_gains": qalys_gains,
            "equipment_maintenance": equipment_maintenance,
            "staff_availability": staff_availability,
            "treatment_fluctuation": treatment_fluctuation,
            "ordered_patient_types": ordered_patient_types,
            "adjacency_matrix": adjacency_matrix,
        }

    def solve(self, instance):
        treatment_costs = instance['treatment_costs']
        equipment_costs = instance['equipment_costs']
        staff_costs = instance['staff_costs']
        qalys_gains = instance['qalys_gains']
        equipment_maintenance = instance['equipment_maintenance']
        staff_availability = instance['staff_availability']
        treatment_fluctuation = instance['treatment_fluctuation']
        ordered_patient_types = instance['ordered_patient_types']
        adjacency_matrix = instance['adjacency_matrix']

        model = Model("HospitalResourceAllocation")
        n_departments = len(treatment_costs)
        n_patient_types = len(equipment_costs[0])
        
        # Define decision variables
        equipment_usage_vars = {d: model.addVar(vtype="B", name=f"EquipmentUsage_{d}") for d in range(n_departments)}
        staff_allocation_vars = {(d, p): model.addVar(vtype="I", name=f"Department_{d}_PatientType_{p}", lb=0) for d in range(n_departments) for p in range(n_patient_types)}
        equipment_maintenance_vars = {d: model.addVar(vtype="C", name=f"EquipmentMaintenance_{d}", lb=0) for d in range(n_departments)}
        staff_usage_vars = {p: model.addVar(vtype="C", name=f"StaffUsage_{p}", lb=0) for p in range(n_patient_types)}
        qalys_vars = {p: model.addVar(vtype="C", name=f"QALYs_{p}", lb=0) for p in range(n_patient_types)}
        flow_vars = {(i, j): model.addVar(vtype="C", name=f"Flow_{i}_{j}", lb=0) for i in range(n_patient_types) for j in range(n_patient_types)}

        # Define objectives
        objectives = []

        # Objective 1: Maximize QALYs
        obj1 = quicksum(qalys_gains[p] * staff_allocation_vars[d, p] * treatment_fluctuation[p] for d in range(n_departments) for p in range(n_patient_types))
        objectives.append(obj1)

        # Objective 2: Minimize treatment and equipment costs
        obj2 = (
            quicksum(treatment_costs[d] * equipment_usage_vars[d] for d in range(n_departments)) +
            quicksum(equipment_costs[d][p] * staff_allocation_vars[d, p] for d in range(n_departments) for p in range(n_patient_types)) +
            quicksum(equipment_maintenance_vars[d] * equipment_maintenance[d] for d in range(n_departments))
        )
        objectives.append(-obj2)

        # Objective 3: Minimize network flow cost to balance load between patient types
        obj3 = quicksum(adjacency_matrix[i][j] * flow_vars[i, j] for i in range(n_patient_types) for j in range(n_patient_types))
        objectives.append(-obj3)

        # Set the combined weighted objective
        total_objective = sum(objectives)
        model.setObjective(total_objective, "maximize")

        # Add constraints
        for p in range(n_patient_types):
            model.addCons(quicksum(staff_allocation_vars[d, p] for d in range(n_departments)) == 1, f"PatientType_{p}_Assignment")
        
        for d in range(n_departments):
            for p in range(n_patient_types):
                model.addCons(staff_allocation_vars[d, p] <= equipment_usage_vars[d], f"Department_{d}_Service_{p}")
        
        for d in range(n_departments):
            model.addCons(quicksum(staff_allocation_vars[d, p] for p in range(n_patient_types)) <= staff_costs[d], f"Department_{d}_Capacity")
        
        for d in range(n_departments):
            model.addCons(equipment_maintenance_vars[d] == quicksum(staff_allocation_vars[d, p] * equipment_maintenance[d] for p in range(n_patient_types)), f"EquipmentMaintenance_{d}")

        for p in range(n_patient_types):
            model.addCons(staff_usage_vars[p] <= staff_availability[p], f"StaffUsage_{p}")

        for p in range(n_patient_types):
            model.addCons(qalys_vars[p] == qalys_gains[p] * treatment_fluctuation[p], f"QALYs_{p}")

        for d in range(n_departments):
            for p in range(n_patient_types):
                model.addCons(staff_allocation_vars[d, p] * treatment_fluctuation[p] <= staff_costs[d], f"TreatmentCapacity_{d}_{p}")

        for i in range(n_patient_types - 1):
            p1 = ordered_patient_types[i]
            p2 = ordered_patient_types[i + 1]
            for d in range(n_departments):
                model.addCons(staff_allocation_vars[d, p1] + staff_allocation_vars[d, p2] <= 1, f"SOS_Constraint_Department_{d}_PatientTypes_{p1}_{p2}")

        # Network flow constraints
        for p in range(n_patient_types):
            model.addCons(quicksum(flow_vars[p, q] for q in range(n_patient_types)) - quicksum(flow_vars[q, p] for q in range(n_patient_types)) == staff_allocation_vars[0, p], f"FlowBalance_{p}")

        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_departments': 40,
        'n_patient_types': 112,
        'min_treatment_cost': 1309,
        'max_treatment_cost': 1405,
        'min_equipment_cost': 324,
        'max_equipment_cost': 1968,
        'min_staff_cost': 924,
        'max_staff_cost': 1012,
    }

    hospital_optimizer = HospitalResourceAllocation(parameters, seed=42)
    instance = hospital_optimizer.generate_instance()
    solve_status, solve_time, objective_value = hospital_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")