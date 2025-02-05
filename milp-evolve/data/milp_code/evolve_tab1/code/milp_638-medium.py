import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class ShiftScheduler:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_airport_graph(self):
        n_nodes = np.random.randint(self.min_airports, self.max_airports)
        G = nx.watts_strogatz_graph(n=n_nodes, k=self.small_world_k, p=self.small_world_p, seed=self.seed)
        return G

    def generate_travel_data(self, G):
        for node in G.nodes:
            G.nodes[node]['workload'] = np.random.randint(20, 200)
        for u, v in G.edges:
            G[u][v]['travel_time'] = np.random.randint(10, 60)
            G[u][v]['cost'] = np.random.randint(20, 200)
    
    def generate_conflict_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.conflict_rate:
                E_invalid.add(edge)
        return E_invalid

    def get_hotel_assignments(self, G):
        assignments = list(nx.find_cliques(G))
        return assignments

    def generate_warehouse_graph(self):
        n_customers = np.random.randint(self.min_warehouses, self.max_warehouses)
        G = nx.erdos_renyi_graph(n=n_customers, p=self.er_prob, directed=True, seed=self.seed)
        return G

    def generate_customers_resources_data(self, G):
        for node in G.nodes:
            G.nodes[node]['demand'] = np.random.randint(1, 100)
        for u, v in G.edges:
            G[u][v]['transport_cost'] = np.random.randint(1, 20)
            G[u][v]['capacity'] = np.random.randint(1, 10)

    def generate_supplier_capacities(self, G):
        supplier_capacity = {node: np.random.randint(100, 1000) for node in G.nodes}
        return supplier_capacity

    def get_instance(self):
        # Generate airport graph and data
        G_airport = self.generate_airport_graph()
        self.generate_travel_data(G_airport)
        E_invalid = self.generate_conflict_data(G_airport)
        hotel_assignments = self.get_hotel_assignments(G_airport)

        hotel_cap = {node: np.random.randint(50, 200) for node in G_airport.nodes}
        travel_costs = {(u, v): np.random.uniform(10.0, 60.0) for u, v in G_airport.edges}
        daily_tasks = [(assignment, np.random.uniform(100, 500)) for assignment in hotel_assignments]

        task_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            task_scenarios[s]['workload'] = {node: np.random.normal(G_airport.nodes[node]['workload'], G_airport.nodes[node]['workload'] * self.work_variation)
                                             for node in G_airport.nodes}
            task_scenarios[s]['travel_time'] = {(u, v): np.random.normal(G_airport[u][v]['travel_time'], G_airport[u][v]['travel_time'] * self.time_variation)
                                                for u, v in G_airport.edges}
            task_scenarios[s]['hotel_cap'] = {node: np.random.normal(hotel_cap[node], hotel_cap[node] * self.cap_variation)
                                              for node in G_airport.nodes}

        delay_penalties = {node: np.random.uniform(20, 100) for node in G_airport.nodes}
        hotel_costs = {(u, v): np.random.uniform(15.0, 90.0) for u, v in G_airport.edges}
        hotel_availability = {node: np.random.uniform(50, 150) for node in G_airport.nodes}

        # Generate warehouse graph and data
        G_warehouse = self.generate_warehouse_graph()
        self.generate_customers_resources_data(G_warehouse)
        supplier_capacity = self.generate_supplier_capacities(G_warehouse)
        supply_demand_complexity = {(u, v): np.random.uniform(0.0, 2.0) for u, v in G_warehouse.edges}
        supply_compatibility = {(u, v): np.random.randint(0, 2) for u, v in G_warehouse.edges}

        warehouse_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            warehouse_scenarios[s]['demand'] = {node: np.random.normal(G_warehouse.nodes[node]['demand'], G_warehouse.nodes[node]['demand'] * self.demand_deviation)
                                            for node in G_warehouse.nodes}
            warehouse_scenarios[s]['transport_cost'] = {(u, v): np.random.normal(G_warehouse[u][v]['transport_cost'], G_warehouse[u][v]['transport_cost'] * self.cost_deviation)
                                                    for u, v in G_warehouse.edges}
            warehouse_scenarios[s]['supplier_capacity'] = {node: np.random.normal(supplier_capacity[node], supplier_capacity[node] * self.capacity_deviation)
                                                        for node in G_warehouse.nodes}

        return {
            'G_airport': G_airport,
            'E_invalid': E_invalid,
            'hotel_assignments': hotel_assignments,
            'hotel_cap': hotel_cap,
            'travel_costs': travel_costs,
            'daily_tasks': daily_tasks,
            'task_scenarios': task_scenarios,
            'delay_penalties': delay_penalties,
            'hotel_costs': hotel_costs,
            'hotel_availability': hotel_availability,
            'G_warehouse': G_warehouse,
            'supplier_capacity': supplier_capacity, 
            'supply_demand_complexity': supply_demand_complexity,
            'supply_compatibility': supply_compatibility,
            'warehouse_scenarios': warehouse_scenarios
        }

    def solve(self, instance):
        G_airport, E_invalid, hotel_assignments = instance['G_airport'], instance['E_invalid'], instance['hotel_assignments']
        hotel_cap = instance['hotel_cap']
        travel_costs = instance['travel_costs']
        daily_tasks = instance['daily_tasks']
        task_scenarios = instance['task_scenarios']
        delay_penalties = instance['delay_penalties']
        hotel_costs = instance['hotel_costs']
        hotel_availability = instance['hotel_availability']
        
        G_warehouse = instance['G_warehouse']
        supplier_capacity = instance['supplier_capacity']
        supply_demand_complexity = instance['supply_demand_complexity']
        supply_compatibility = instance['supply_compatibility']
        warehouse_scenarios = instance['warehouse_scenarios']
        
        model = Model("ShiftSchedulerWithWarehouse")

        # Define variables for airport shifts and routes
        shift_vars = {node: model.addVar(vtype="B", name=f"Shift_{node}") for node in G_airport.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G_airport.edges}
        delay_var = model.addVar(vtype="C", name="Total_Delay")
        daily_task_vars = {i: model.addVar(vtype="B", name=f"Task_{i}") for i in range(len(daily_tasks))}
        hotel_vars = {node: model.addVar(vtype="C", name=f"Hotel_{node}") for node in G_airport.nodes}

        # Define variables for warehouses and supply routes
        warehouse_vars = {f"w{node}": model.addVar(vtype="B", name=f"w{node}") for node in G_warehouse.nodes}
        supply_vars = {f"s{u}_{v}": model.addVar(vtype="B", name=f"s{u}_{v}") for u, v in G_warehouse.edges}
        
        # Scenario-specific variables
        demand_vars = {s: {f"c{node}_s{s}": model.addVar(vtype="C", name=f"c{node}_s{s}") for node in G_warehouse.nodes} for s in range(self.no_of_scenarios)}
        transport_cost_vars = {s: {f"t{u}_{v}_s{s}": model.addVar(vtype="C", name=f"t{u}_{v}_s{s}") for u, v in G_warehouse.edges} for s in range(self.no_of_scenarios)}
        capacity_vars = {s: {f"cap{node}_s{s}": model.addVar(vtype="C", name=f"cap{node}_s{s}") for node in G_warehouse.nodes} for s in range(self.no_of_scenarios)}

        # Objective function
        objective_expr = quicksum(
            task_scenarios[s]['workload'][node] * shift_vars[node]
            for s in range(self.no_of_scenarios) for node in G_airport.nodes
        )
        objective_expr -= quicksum(
            travel_costs[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G_airport.edges
        )
        objective_expr += quicksum(cost * daily_task_vars[i] for i, (assignment, cost) in enumerate(daily_tasks))
        objective_expr -= quicksum(delay_penalties[node] * delay_var for node in G_airport.nodes)
        objective_expr -= quicksum(hotel_costs[(u, v)] * route_vars[f"Route_{u}_{v}"] for u, v in G_airport.edges)
        objective_expr += quicksum(hotel_vars[node] for node in G_airport.nodes)
        
        objective_expr += quicksum(
            warehouse_scenarios[s]['demand'][node] * demand_vars[s][f"c{node}_s{s}"]
            for s in range(self.no_of_scenarios) for node in G_warehouse.nodes
        )
        objective_expr -= quicksum(
            warehouse_scenarios[s]['transport_cost'][(u, v)] * transport_cost_vars[s][f"t{u}_{v}_s{s}"]
            for s in range(self.no_of_scenarios) for u, v in G_warehouse.edges
        )
        objective_expr -= quicksum(
            warehouse_scenarios[s]['supplier_capacity'][node] * warehouse_scenarios[s]['demand'][node]
            for s in range(self.no_of_scenarios) for node in G_warehouse.nodes
        )
        objective_expr -= quicksum(
            supply_demand_complexity[(u, v)] * supply_vars[f"s{u}_{v}"]
            for u, v in G_warehouse.edges
        )
        objective_expr -= quicksum(
            supply_compatibility[(u, v)] * supply_vars[f"s{u}_{v}"]
            for u, v in G_warehouse.edges
        )

        # Constraints

        # Airport constraints
        for i, assignment in enumerate(hotel_assignments):
            model.addCons(
                quicksum(shift_vars[node] for node in assignment) <= 1,
                name=f"ShiftGroup_{i}"
            )

        for u, v in G_airport.edges:
            model.addCons(
                shift_vars[u] + shift_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"Flow_{u}_{v}"
            )
        
        for node in G_airport.nodes:
            model.addCons(
                hotel_vars[node] <= hotel_availability[node],
                name=f"Hotel_Limit_{node}"
            )

        model.addCons(
            delay_var <= self.max_delay,
            name="Max_Delay"
        )

        # Warehouse constraints
        for node in G_warehouse.nodes:
            model.addCons(
                quicksum(demand_vars[s][f"c{node}_s{s}"] for s in range(self.no_of_scenarios)) <= warehouse_vars[f"w{node}"] * supplier_capacity[node],
                name=f"Capacity_{node}"
            )

        for u, v in G_warehouse.edges:
            model.addCons(
                warehouse_vars[f"w{u}"] + warehouse_vars[f"w{v}"] <= 1 + supply_vars[f"s{u}_{v}"],
                name=f"Sup1_{u}_{v}"
            )
            model.addCons(
                warehouse_vars[f"w{u}"] + warehouse_vars[f"w{v}"] >= 2 * supply_vars[f"s{u}_{v}"],
                name=f"Sup2_{u}_{v}"
            )

        weekly_budget = model.addVar(vtype="C", name="weekly_budget")
        model.addCons(
            weekly_budget <= self.weekly_budget_limit,
            name="Weekly_budget_limit"
        )
        
        # Robust constraints to ensure feasibility across all scenarios
        for s in range(self.no_of_scenarios):
            for node in G_warehouse.nodes:
                model.addCons(
                    demand_vars[s][f"c{node}_s{s}"] == weekly_budget, # Ensure demand acts within budget
                    name=f"RobustDemand_{node}_s{s}"
                )
                model.addCons(
                    capacity_vars[s][f"cap{node}_s{s}"] == warehouse_vars[f"w{node}"],
                    name=f"RobustCapacity_{node}_s{s}"
                )
            for u, v in G_warehouse.edges:
                model.addCons(
                    transport_cost_vars[s][f"t{u}_{v}_s{s}"] == supply_vars[f"s{u}_{v}"],
                    name=f"RobustTransportCost_{u}_{v}_s{s}"
                )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_airports': 15,
        'max_airports': 500,
        'conflict_rate': 0.1,
        'max_delay': 24,
        'no_of_scenarios': 300,
        'work_variation': 0.8,
        'time_variation': 0.52,
        'cap_variation': 0.38,
        'financial_param1': 300,
        'financial_param2': 400,
        'travel_cost_param_1': 187,
        'zero_hour_contracts': 35,
        'min_employee_shifts': 0,
        'max_employee_shifts': 150,
        'small_world_k': 1,
        'small_world_p': 0.73,
        'min_customers': 27,
        'max_customers': 153,
        'er_prob': 0.38,
        'weekly_budget_limit': 1344,
        'demand_deviation': 0.24,
        'cost_deviation': 0.59,
        'capacity_deviation': 0.52,
        'min_warehouses': 25,
        'max_warehouses': 150,
    }

    scheduler = ShiftScheduler(parameters, seed=seed)
    instance = scheduler.get_instance()
    solve_status, solve_time = scheduler.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")