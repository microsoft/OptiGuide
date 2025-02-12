import random
import time
import numpy as np
from itertools import combinations
from pyscipopt import Model, quicksum
import networkx as nx

class Graph:
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.nodes = np.arange(number_of_nodes)
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    @staticmethod
    def erdos_renyi(number_of_nodes, probability, seed=None):
        graph = nx.erdos_renyi_graph(number_of_nodes, probability, seed=seed)
        edges = set(graph.edges)
        degrees = [d for (n, d) in graph.degree]
        neighbors = {node: set(graph.neighbors(node)) for node in graph.nodes}
        return Graph(number_of_nodes, edges, degrees, neighbors)

class FoodSupplyChainOptimization:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_instance(self):
        center_costs = np.random.randint(self.min_center_cost, self.max_center_cost + 1, self.n_centers)
        store_costs = np.random.randint(self.min_task_cost, self.max_task_cost + 1, (self.n_centers, self.n_tasks))
        capacities = np.random.randint(self.min_center_capacity, self.max_center_capacity + 1, self.n_centers)
        demands = np.random.randint(1, 10, self.n_tasks)

        graph = Graph.erdos_renyi(self.n_centers, self.link_probability, seed=self.seed)
        edge_weights = np.random.randint(1, 10, size=len(graph.edges))

        transport_costs = np.random.randint(1, 20, (self.n_centers, self.n_centers))
        transport_capacities = np.random.randint(self.flow_capacity_min, self.flow_capacity_max + 1, (self.n_centers, self.n_centers))

        shifts = np.random.randint(1, self.max_shifts + 1)
        shift_capacity = np.random.randint(1, self.max_shift_capacity + 1, size=shifts)
        inventory_holding_costs = np.random.randint(1, 10, self.n_centers)
        renewable_energy_costs = np.random.randint(1, 10, (self.n_centers, self.n_centers))
        carbon_emissions = np.random.randint(1, 10, (self.n_centers, self.n_centers))
        
        water_usage_limits = np.random.randint(50, 500, self.n_centers)
        waste_disposal_limits = np.random.randint(50, 500, self.n_centers)
        
        raw_material_transport_costs = np.random.randint(10, 50, (self.n_centers, self.n_centers))
        finished_product_transport_costs = np.random.randint(10, 50, (self.n_centers, self.n_centers))
        
        energy_consumption_costs = np.random.randint(5, 30, (self.n_centers, self.n_centers))
        
        return {
            "center_costs": center_costs,
            "store_costs": store_costs,
            "capacities": capacities,
            "demands": demands,
            "graph": graph,
            "edge_weights": edge_weights,
            "transport_costs": transport_costs,
            "transport_capacities": transport_capacities,
            "shifts": shifts,
            "shift_capacity": shift_capacity,
            "inventory_holding_costs": inventory_holding_costs,
            "renewable_energy_costs": renewable_energy_costs,
            "carbon_emissions": carbon_emissions,
            "water_usage_limits": water_usage_limits,
            "waste_disposal_limits": waste_disposal_limits,
            "raw_material_transport_costs": raw_material_transport_costs,
            "finished_product_transport_costs": finished_product_transport_costs,
            "energy_consumption_costs": energy_consumption_costs,
        }

    def solve(self, instance):
        center_costs = instance['center_costs']
        store_costs = instance['store_costs']
        capacities = instance['capacities']
        demands = instance['demands']
        graph = instance['graph']
        edge_weights = instance['edge_weights']
        transport_costs = instance['transport_costs']
        transport_capacities = instance['transport_capacities']
        shifts = instance['shifts']
        shift_capacity = instance['shift_capacity']
        inventory_holding_costs = instance['inventory_holding_costs']
        renewable_energy_costs = instance['renewable_energy_costs']
        carbon_emissions = instance['carbon_emissions']
        water_usage_limits = instance['water_usage_limits']
        waste_disposal_limits = instance['waste_disposal_limits']
        raw_material_transport_costs = instance['raw_material_transport_costs']
        finished_product_transport_costs = instance['finished_product_transport_costs']
        energy_consumption_costs = instance['energy_consumption_costs']

        model = Model("FoodSupplyChainOptimization")
        n_centers = len(center_costs)
        n_tasks = len(store_costs[0])

        alloc_vars = {c: model.addVar(vtype="B", name=f"Center_Allocation_{c}") for c in range(n_centers)}
        task_vars = {(c, t): model.addVar(vtype="B", name=f"Center_Task_{c}_Task_{t}") for c in range(n_centers) for t in range(n_tasks)}
        usage_vars = {(i, j): model.addVar(vtype="I", name=f"Usage_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}
        flow_vars = {(i, j): model.addVar(vtype="I", name=f"Flow_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}

        shift_vars = {f"shift_{m}_{s}": model.addVar(vtype="B", name=f"Shift_{m}_{s}") for m in range(n_centers) for s in range(shifts)}
        inventory_level = {f"inv_{node}": model.addVar(vtype="C", name=f"Inventory_{node}") for node in range(n_centers)}
        renewable_energy_vars = {(i, j): model.addVar(vtype="B", name=f"RenewableEnergy_{i}_{j}") for (i, j) in graph.edges}
        
        water_usage_vars = {f"water_{m}": model.addVar(vtype="C", name=f"Water_Usage_{m}") for m in range(n_centers)}
        waste_disposal_vars = {f"waste_{m}": model.addVar(vtype="C", name=f"Waste_Disposal_{m}") for m in range(n_centers)}
        raw_material_transport_vars = {(i, j): model.addVar(vtype="I", name=f"Raw_Material_Transport_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}
        finished_product_transport_vars = {(i, j): model.addVar(vtype="I", name=f"Finished_Product_Transport_{i}_{j}") for i in range(n_centers) for j in range(n_centers)}
        energy_vars = {f"energy_{m}": model.addVar(vtype="C", name=f"Energy_Consumption_{m}") for m in range(n_centers)}

        model.setObjective(
            quicksum(center_costs[c] * alloc_vars[c] for c in range(n_centers)) +
            quicksum(store_costs[c, t] * task_vars[c, t] for c in range(n_centers) for t in range(n_tasks)) +
            quicksum(transport_costs[i, j] * flow_vars[i, j] for i in range(n_centers) for j in range(n_centers)) +
            quicksum(inventory_holding_costs[node] * inventory_level[f"inv_{node}"] for node in range(n_centers)) +
            quicksum(renewable_energy_costs[i, j] * renewable_energy_vars[(i, j)] for (i, j) in graph.edges) +
            quicksum(carbon_emissions[i, j] * flow_vars[i, j] for (i, j) in graph.edges) +
            quicksum(raw_material_transport_costs[i, j] * raw_material_transport_vars[(i, j)] for i in range(n_centers) for j in range(n_centers)) +
            quicksum(finished_product_transport_costs[i, j] * finished_product_transport_vars[(i, j)] for i in range(n_centers) for j in range(n_centers)) +
            quicksum(energy_consumption_costs[i, j] * flow_vars[i, j] for i in range(n_centers) for j in range(n_centers)), 
            "minimize"
        )

        for t in range(n_tasks):
            model.addCons(quicksum(task_vars[c, t] for c in range(n_centers)) == 1, f"Task_{t}_Allocation")

        for c in range(n_centers):
            for t in range(n_tasks):
                model.addCons(task_vars[c, t] <= alloc_vars[c], f"Center_{c}_Serve_{t}")

        for c in range(n_centers):
            model.addCons(quicksum(demands[t] * task_vars[c, t] for t in range(n_tasks)) <= capacities[c], f"Center_{c}_Capacity")

        for edge in graph.edges:
            model.addCons(alloc_vars[edge[0]] + alloc_vars[edge[1]] <= 1, f"Edge_{edge[0]}_{edge[1]}")

        for i in range(n_centers):
            model.addCons(
                quicksum(usage_vars[i, j] for j in range(n_centers) if i != j) ==
                quicksum(usage_vars[j, i] for j in range(n_centers) if i != j),
                f"Usage_Conservation_{i}"
            )

        for j in range(n_centers):
            model.addCons(
                quicksum(flow_vars[i, j] for i in range(n_centers) if i != j) ==
                quicksum(task_vars[j, t] for t in range(n_tasks)),
                f"Flow_Conservation_{j}"
            )

        for i in range(n_centers):
            for j in range(n_centers):
                if i != j:
                    model.addCons(flow_vars[i, j] <= transport_capacities[i, j], f"Flow_Capacity_{i}_{j}")

        for m in range(n_centers):
            model.addCons(
                quicksum(shift_vars[f"shift_{m}_{s}"] for s in range(shifts)) <= 1,
                f"Shift_Allocation_{m}"
            )

        for s in range(shifts):
            model.addCons(
                quicksum(shift_vars[f"shift_{m}_{s}"] for m in range(n_centers)) <= shift_capacity[s],
                f"Shift_Capacity_{s}"
            )

        for (i, j) in graph.edges:
            model.addCons(
                flow_vars[i, j] <= renewable_energy_vars[(i, j)] * 10,
                name=f"RenewableEnergy_Limit_{i}_{j}"
            )
            
        for m in range(n_centers):
            model.addCons(
                water_usage_vars[f"water_{m}"] <= water_usage_limits[m],
                name=f"Water_Usage_Limit_{m}"
            )
            model.addCons(
                waste_disposal_vars[f"waste_{m}"] <= waste_disposal_limits[m],
                name=f"Waste_Disposal_Limit_{m}"
            )

        for m in range(n_centers):
            model.addCons(
                energy_vars[f"energy_{m}"] <= quicksum(demands[t] * task_vars[m, t] for t in range(n_tasks)),
                name=f"Energy_Consumption_Limit_{m}"
            )

        for i in range(n_centers):
            for j in range(n_centers):
                if i != j:
                    model.addCons(
                        raw_material_transport_vars[(i, j)] >= 0,
                        name=f"Raw_Material_Transport_{i}_{j}_Nonnegative"
                    )
                    model.addCons(
                        finished_product_transport_vars[(i, j)] >= 0,
                        name=f"Finished_Product_Transport_{i}_{j}_Nonnegative"
                    )

        start_time = time.time()
        result = model.optimize()
        end_time = time.time()

        return model.getStatus(), end_time - start_time, model.getObjVal()

if __name__ == '__main__':
    seed = 42
    parameters = {
        'n_centers': 22,
        'n_tasks': 336,
        'min_task_cost': 1653,
        'max_task_cost': 3000,
        'min_center_cost': 1397,
        'max_center_cost': 5000,
        'min_center_capacity': 1268,
        'max_center_capacity': 1518,
        'link_probability': 0.31,
        'flow_capacity_min': 937,
        'flow_capacity_max': 3000,
        'max_shifts': 2058,
        'max_shift_capacity': 35,
    }

    resource_optimizer = FoodSupplyChainOptimization(parameters, seed=seed)
    instance = resource_optimizer.generate_instance()
    solve_status, solve_time, objective_value = resource_optimizer.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")