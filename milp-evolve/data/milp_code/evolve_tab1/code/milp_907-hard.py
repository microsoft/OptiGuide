import random
import time
import numpy as np
from itertools import combinations
import networkx as nx
from pyscipopt import Model, quicksum

class SCND:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    ################# Data Generation #################
    def generate_random_graph(self):
        G = nx.barabasi_albert_graph(n=self.n_nodes, m=self.ba_edges, seed=self.seed)
        capacities = np.random.uniform(self.cap_min, self.cap_max, size=(self.n_nodes, self.n_nodes))
        transport_costs = np.random.uniform(self.tc_min, self.tc_max, size=(self.n_nodes, self.n_nodes))
        return G, capacities, transport_costs

    def generate_demand(self):
        demands = np.random.uniform(self.demand_min, self.demand_max, size=self.n_nodes)
        return demands

    def generate_facilities(self):
        facilities = np.random.uniform(self.facility_min, self.facility_max, size=self.n_nodes)
        opening_costs = np.random.uniform(self.opening_cost_min, self.opening_cost_max, size=self.n_nodes)
        maintenance_costs = np.random.uniform(self.maintenance_cost_min, self.maintenance_cost_max, size=self.n_nodes)
        equipment_lifespans = np.random.randint(self.equipment_lifespan_min, self.equipment_lifespan_max, size=self.n_nodes)
        return facilities, opening_costs, maintenance_costs, equipment_lifespans

    def generate_environmental_impacts(self):
        environmental_costs = np.random.uniform(self.env_cost_min, self.env_cost_max, size=(self.n_nodes, self.n_nodes))
        return environmental_costs

    def generate_storage_capacities(self):
        storage_capacities = [random.randint(50, 200) for _ in range(self.n_nodes)]
        return storage_capacities

    def generate_inventory_levels(self):
        inventory_levels = [random.randint(10, 150) for _ in range(self.n_nodes)]
        return inventory_levels

    def generate_instance(self):
        self.n_nodes = np.random.randint(self.min_n_nodes, self.max_n_nodes+1)
        G, capacities, transport_costs = self.generate_random_graph()
        demands = self.generate_demand()
        facilities, opening_costs, maintenance_costs, equipment_lifespans = self.generate_facilities()
        environmental_costs = self.generate_environmental_impacts()
        storage_capacities = self.generate_storage_capacities()
        inventory_levels = self.generate_inventory_levels()

        res = {
            'graph': G,
            'capacities': capacities,
            'transport_costs': transport_costs,
            'demands': demands,
            'facilities': facilities,
            'opening_costs': opening_costs,
            'maintenance_costs': maintenance_costs,
            'equipment_lifespans': equipment_lifespans,
            'environmental_costs': environmental_costs,
            'storage_capacities': storage_capacities,
            'inventory_levels': inventory_levels
        }

        # Additional new data for batching and maintenance schedule
        self.batch_sizes = [random.randint(1, 10) for _ in range(self.n_nodes)]
        self.maintenance_schedule = [random.choice([0, 1]) for _ in range(self.n_nodes)]

        res['batch_sizes'] = self.batch_sizes
        res['maintenance_schedule'] = self.maintenance_schedule

        return res

    ################# PySCIPOpt Modeling #################
    def solve(self, instance):
        G = instance['graph']
        capacities = instance['capacities']
        transport_costs = instance['transport_costs']
        demands = instance['demands']
        facilities = instance['facilities']
        opening_costs = instance['opening_costs']
        maintenance_costs = instance['maintenance_costs']
        equipment_lifespans = instance['equipment_lifespans']
        environmental_costs = instance['environmental_costs']
        batch_sizes = instance['batch_sizes']
        maintenance_schedule = instance['maintenance_schedule']
        storage_capacities = instance['storage_capacities']
        inventory_levels = instance['inventory_levels']

        model = Model("SCND")
        Facility_Open = {i: model.addVar(vtype="B", name=f"Facility_Open_{i}") for i in range(self.n_nodes)}
        Allocation = {(i, j): model.addVar(vtype="C", name=f"Allocation_{i}_{j}") for i in range(self.n_nodes) for j in range(self.n_nodes)}
        Transport_Capacity = {(i, j): model.addVar(vtype="C", name=f"Transport_Capacity_{i}_{j}") for i in range(self.n_nodes) for j in range(self.n_nodes)}

        Batch_Size = {(i, j): model.addVar(vtype="I", name=f"Batch_Size_{i}_{j}") for i in range(self.n_nodes) for j in range(self.n_nodes)}
        Queue_Time = {(i, j): model.addVar(vtype="C", name=f"Queue_Time_{i}_{j}") for i in range(self.n_nodes) for j in range(self.n_nodes)}
        Maintenance = {i: model.addVar(vtype="B", name=f"Maintenance_{i}") for i in range(self.n_nodes)}

        Temp_Control = {(i, j): model.addVar(vtype="B", name=f"Temp_Control_{i}_{j}") for i, j in G.edges}

        # Objective function
        objective_expr = quicksum(
            opening_costs[i] * Facility_Open[i]
            for i in range(self.n_nodes)
        ) + quicksum(
            Allocation[i, j] * transport_costs[i, j]
            for i in range(self.n_nodes) for j in range(self.n_nodes)
        ) + quicksum(
            environmental_costs[i, j] * Transport_Capacity[i, j]
            for i in range(self.n_nodes) for j in range(self.n_nodes)
        ) + quicksum(
            maintenance_costs[i] * Facility_Open[i]
            for i in range(self.n_nodes)
        ) + quicksum(
            batch_sizes[i] * Queue_Time[i, j]
            for i in range(self.n_nodes) for j in range(self.n_nodes)
        ) + quicksum(
            Temp_Control[i, j] * self.temp_control_cost
            for i, j in G.edges
        )

        model.setObjective(objective_expr, "minimize")

        # Constraints
        for i in range(self.n_nodes):
            # Facility capacity constraint
            model.addCons(
                quicksum(Allocation[i, j] for j in range(self.n_nodes)) <= facilities[i] * Facility_Open[i],
                f"Facility_Capacity_{i}"
            )
            # Demand satisfaction constraint
            model.addCons(
                quicksum(Allocation[j, i] for j in range(self.n_nodes)) == demands[i],
                f"Demand_Satisfaction_{i}"
            )
            # Equipment lifespan constraint
            model.addCons(
                quicksum(Allocation[i, j] for j in range(self.n_nodes)) <= equipment_lifespans[i],
                f"Equipment_Lifespan_{i}"
            )
            # Inventory storage constraints
            model.addCons(
                Facility_Open[i] * inventory_levels[i] <= storage_capacities[i],
                f"Storage_Capacity_{i}"
            )

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                # Transport capacity constraint
                model.addCons(
                    Allocation[i, j] <= capacities[i, j] * Transport_Capacity[i, j],
                    f"Transport_Capacity_{i}_{j}"
                )
                # Environmental impact constraint
                model.addCons(
                    environmental_costs[i, j] * Transport_Capacity[i, j] <= self.env_threshold,
                    f"Env_Impact_{i}_{j}"
                )
                # Batch size constraints
                model.addCons(
                    Batch_Size[i, j] <= len(batch_sizes),
                    f"Batch_Constraint_{i}_{j}"
                )
                # Queue time constraints
                model.addCons(
                    Queue_Time[i, j] <= 5.0,
                    f"Queue_Time_Constraint_{i}_{j}"
                )

        # Maintenance constraints
        for i in range(self.n_nodes):
            model.addCons(
                Maintenance[i] == maintenance_schedule[i],
                f"Maintenance_Constraint_{i}"
            )
            model.addCons(
                quicksum(Transport_Capacity[i, j] for j in range(self.n_nodes)) <= 10,
                f"Production_Run_Constraint_{i}"
            )

        # Temperature control constraints
        for i, j in G.edges:
            model.addCons(
                Temp_Control[i, j] >= (Facility_Open[i] + Facility_Open[j] - 1),
                f"Temp_Control_{i}_{j}"
            )
        
        # At least 30% of edges must have temperature control
        model.addCons(
            quicksum(Temp_Control[i, j] for i, j in G.edges) >= 0.3 * len(G.edges),
            "Min_Temp_Control"
        )
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time


if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n_nodes': 40,
        'max_n_nodes': 600,
        'ba_edges': 60,
        'facility_min': 2400,
        'facility_max': 1500,
        'opening_cost_min': 5000,
        'opening_cost_max': 20000,
        'cap_min': 87,
        'cap_max': 20,
        'tc_min': 2,
        'tc_max': 2500,
        'demand_min': 300,
        'demand_max': 20,
        'maintenance_cost_min': 25,
        'maintenance_cost_max': 50,
        'equipment_lifespan_min': 75,
        'equipment_lifespan_max': 500,
        'env_cost_min': 10,
        'env_cost_max': 25,
        'env_threshold': 500,
    }
    parameters.update({
        'batch_size_max': 10,
        'queue_time_max': 5.0,
        'temp_control_cost': 100,
    })

    scnd = SCND(parameters, seed=seed)
    instance = scnd.generate_instance()
    solve_status, solve_time = scnd.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")