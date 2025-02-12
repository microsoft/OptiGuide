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
            G[u][v]['max_travel_time'] = np.random.randint(30, 120)  # New

    def generate_conflict_data(self, G):
        E_invalid = set()
        for edge in G.edges:
            if np.random.random() <= self.conflict_rate:
                E_invalid.add(edge)
        return E_invalid

    def get_hotel_assignments(self, G):
        assignments = list(nx.find_cliques(G))
        return assignments

    def get_instance(self):
        G = self.generate_airport_graph()
        self.generate_travel_data(G)
        E_invalid = self.generate_conflict_data(G)
        hotel_assignments = self.get_hotel_assignments(G)

        hotel_cap = {node: np.random.randint(50, 200) for node in G.nodes}
        travel_costs = {(u, v): np.random.uniform(10.0, 60.0) for u, v in G.edges}
        daily_tasks = [(assignment, np.random.uniform(100, 500)) for assignment in hotel_assignments]

        task_scenarios = [{} for _ in range(self.no_of_scenarios)]
        for s in range(self.no_of_scenarios):
            task_scenarios[s]['workload'] = {node: np.random.normal(G.nodes[node]['workload'], G.nodes[node]['workload'] * self.work_variation)
                                             for node in G.nodes}
            task_scenarios[s]['travel_time'] = {(u, v): np.random.normal(G[u][v]['travel_time'], G[u][v]['travel_time'] * self.time_variation)
                                                for u, v in G.edges}
            task_scenarios[s]['hotel_cap'] = {node: np.random.normal(hotel_cap[node], hotel_cap[node] * self.cap_variation)
                                              for node in G.nodes}

        delay_penalties = {node: np.random.uniform(20, 100) for node in G.nodes}
        hotel_costs = {(u, v): np.random.uniform(15.0, 90.0) for u, v in G.edges}
        hotel_stay = {(u, v): np.random.uniform(30.0, 120.0) for u, v in G.edges}  # New
        hotel_availability = {node: np.random.uniform(50, 150) for node in G.nodes}

        return {
            'G': G,
            'E_invalid': E_invalid,
            'hotel_assignments': hotel_assignments,
            'hotel_cap': hotel_cap,
            'travel_costs': travel_costs,
            'daily_tasks': daily_tasks,
            'task_scenarios': task_scenarios,
            'delay_penalties': delay_penalties,
            'hotel_costs': hotel_costs,
            'hotel_stay': hotel_stay,  # New
            'hotel_availability': hotel_availability
        }

    def solve(self, instance):
        G, E_invalid, hotel_assignments = instance['G'], instance['E_invalid'], instance['hotel_assignments']
        hotel_cap = instance['hotel_cap']
        travel_costs = instance['travel_costs']
        daily_tasks = instance['daily_tasks']
        task_scenarios = instance['task_scenarios']
        delay_penalties = instance['delay_penalties']
        hotel_costs = instance['hotel_costs']
        hotel_stay = instance['hotel_stay']  # New
        hotel_availability = instance['hotel_availability']

        model = Model("ShiftScheduler")

        # Define variables
        shift_vars = {node: model.addVar(vtype="B", name=f"Shift_{node}") for node in G.nodes}
        route_vars = {f"Route_{u}_{v}": model.addVar(vtype="B", name=f"Route_{u}_{v}") for u, v in G.edges}
        delay_var = model.addVar(vtype="C", name="Total_Delay")
        daily_task_vars = {i: model.addVar(vtype="B", name=f"Task_{i}") for i in range(len(daily_tasks))}
        hotel_vars = {node: model.addVar(vtype="C", name=f"Hotel_{node}") for node in G.nodes}

        # Objective function
        objective_expr = quicksum(
            task_scenarios[s]['workload'][node] * shift_vars[node]
            for s in range(self.no_of_scenarios) for node in G.nodes
        )
        objective_expr -= quicksum(
            travel_costs[(u, v)] * route_vars[f"Route_{u}_{v}"]
            for u, v in G.edges
        )
        objective_expr += quicksum(cost * daily_task_vars[i] for i, (assignment, cost) in enumerate(daily_tasks))
        objective_expr -= quicksum(delay_penalties[node] * delay_var for node in G.nodes)
        objective_expr -= quicksum(hotel_costs[(u, v)] * route_vars[f"Route_{u}_{v}"] for u, v in G.edges)
        objective_expr += quicksum(hotel_vars[node] for node in G.nodes)

        # Constraints
        for i, assignment in enumerate(hotel_assignments):
            model.addCons(
                quicksum(shift_vars[node] for node in assignment) <= 1,
                name=f"ShiftGroup_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                shift_vars[u] + shift_vars[v] <= 1 + route_vars[f"Route_{u}_{v}"],
                name=f"Flow_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                hotel_vars[node] <= hotel_availability[node],
                name=f"Hotel_Limit_{node}"
            )

        for u, v in G.edges:
            model.addCons(
                route_vars[f"Route_{u}_{v}"] * G[u][v]['travel_time'] <= G[u][v]['max_travel_time'],
                name=f"Max_TravelTime_{u}_{v}"
            )
            
        for u, v in G.edges:
            model.addCons(
                route_vars[f"Route_{u}_{v}"] * hotel_stay[(u, v)] <= hotel_vars[u] + hotel_vars[v],
                name=f"Max_HotelStay_{u}_{v}"
            )

        model.addCons(
            delay_var <= self.max_delay,
            name="Max_Delay"
        )

        model.setObjective(objective_expr, "maximize")

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_airports': 150,
        'max_airports': 500,
        'conflict_rate': 0.6,
        'max_delay': 240,
        'no_of_scenarios': 100,
        'work_variation': 0.25,
        'time_variation': 0.3,
        'cap_variation': 0.2,
        'financial_param1': 300,
        'financial_param2': 400,
        'travel_cost_param_1': 250,
        'zero_hour_contracts': 700,
        'min_employee_shifts': 5,
        'max_employee_shifts': 50,
        'small_world_k': 20,
        'small_world_p': 0.2,
    }

    scheduler = ShiftScheduler(parameters, seed=seed)
    instance = scheduler.get_instance()
    solve_status, solve_time = scheduler.solve(instance)
    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")