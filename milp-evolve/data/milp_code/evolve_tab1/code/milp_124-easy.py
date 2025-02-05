import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class GISP:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)

        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_random_graph(self):
        n_nodes = np.random.randint(self.min_n, self.max_n)
        G = nx.erdos_renyi_graph(n=n_nodes, p=self.er_prob, seed=self.seed)
        return G

    def generate_revenues_costs(self, G):
        if self.set_type == 'SET1':
            for node in G.nodes:
                G.nodes[node]['revenue'] = np.random.randint(1, 100)
            for u, v in G.edges:
                G[u][v]['cost'] = (G.nodes[u]['revenue'] + G.nodes[v]['revenue']) / float(self.set_param)
        elif self.set_type == 'SET2':
            for node in G.nodes:
                G.nodes[node]['revenue'] = float(self.set_param)
            for u, v in G.edges:
                G[u][v]['cost'] = 1.0

    def generate_revenues_costs_meal(self, n_meals, n_nutrients):
        meals = {f'm_{m}': np.random.randint(0, 100, size=n_nutrients).tolist() for m in range(n_meals)}
        costs = np.random.randint(5, 15, size=n_meals).tolist()
        required_nutrients = np.random.randint(20, 80, size=n_nutrients).tolist()
        return meals, costs, required_nutrients

    def generate_removable_edges(self, G):
        E2 = set()
        for edge in G.edges:
            if np.random.random() <= self.alpha:
                E2.add(edge)
        return E2

    def find_maximal_cliques(self, G):
        cliques = list(nx.find_cliques(G))
        return cliques

    def generate_instance(self):
        G = self.generate_random_graph()
        self.generate_revenues_costs(G)
        E2 = self.generate_removable_edges(G)
        cliques = self.find_maximal_cliques(G)
        
        inventory_holding_costs = {node: np.random.randint(1, 10) for node in G.nodes}
        renewable_energy_costs = {(u, v): np.random.randint(1, 10) for u, v in G.edges}
        carbon_emissions = {(u, v): np.random.randint(1, 10) for u, v in G.edges}
        weather_conditions = {(u, v): np.random.uniform(0.5, 1.5) for u, v in G.edges}
        terrain_difficulty = {(u, v): np.random.uniform(1, 3) for u, v in G.edges}
        
        n_meals = np.random.randint(self.min_meals, self.max_meals + 1)
        n_nutrients = np.random.randint(self.min_nutrients, self.max_nutrients)
        meals, meal_cost, required_nutrients = self.generate_revenues_costs_meal(n_meals, n_nutrients)

        return {'G': G, 'E2': E2, 'cliques': cliques, 
                'inventory_holding_costs': inventory_holding_costs, 
                'renewable_energy_costs': renewable_energy_costs,
                'carbon_emissions': carbon_emissions,
                'weather_conditions': weather_conditions,
                'terrain_difficulty': terrain_difficulty,
                'meals': meals,
                'meal_cost': meal_cost,
                'required_nutrients': required_nutrients,
                'n_meals': n_meals,
                'n_nutrients': n_nutrients}

    def solve(self, instance):
        G, E2, cliques = instance['G'], instance['E2'], instance['cliques']
        inventory_holding_costs = instance['inventory_holding_costs']
        renewable_energy_costs = instance['renewable_energy_costs']
        carbon_emissions = instance['carbon_emissions']
        weather_conditions = instance['weather_conditions']
        terrain_difficulty = instance['terrain_difficulty']
        meals = instance['meals']
        meal_cost = instance['meal_cost']
        required_nutrients = instance['required_nutrients']
        n_meals = instance['n_meals']
        n_nutrients = instance['n_nutrients']
        
        model = Model("GISP")
        node_vars = {f"x{node}":  model.addVar(vtype="B", name=f"x{node}") for node in G.nodes}
        edge_vars = {f"y{u}_{v}": model.addVar(vtype="B", name=f"y{u}_{v}") for u, v in G.edges}
        piecewise_vars = {f"z{u}_{v}": model.addVar(vtype="B", name=f"z{u}_{v}") for u, v in G.edges}
        
        energy_vars = {f"e{node}": model.addVar(vtype="C", name=f"e{node}") for node in G.nodes}
        yearly_budget = model.addVar(vtype="C", name="yearly_budget")

        demand_level = {f"dem_{node}": model.addVar(vtype="C", name=f"dem_{node}") for node in G.nodes}
        renewable_energy_vars = {f"re_{u}_{v}": model.addVar(vtype="B", name=f"re_{u}_{v}") for u, v in G.edges}

        meal_vars = {f"m_{m}": model.addVar(vtype="B", name=f"m_{m}") for m in range(n_meals)}

        objective_expr = quicksum(
            G.nodes[node]['revenue'] * node_vars[f"x{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            G[u][v]['cost'] * edge_vars[f"y{u}_{v}"]
            for u, v in E2
        )

        objective_expr -= quicksum(
            inventory_holding_costs[node] * energy_vars[f"e{node}"]
            for node in G.nodes
        )

        objective_expr -= quicksum(
            renewable_energy_costs[(u, v)] * renewable_energy_vars[f"re_{u}_{v}"]
            for u, v in G.edges
        )

        objective_expr -= quicksum(
            carbon_emissions[(u, v)] * edge_vars[f"y{u}_{v}"]
            for u, v in G.edges
        )

        # Add meal cost to the objective function
        objective_expr -= quicksum(
            meal_cost[m] * meal_vars[f"m_{m}"]
            for m in range(n_meals)
        )

        # Original constraints from GISP model
        for u, v in G.edges:
            if (u, v) in E2:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] - edge_vars[f"y{u}_{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )
            else:
                model.addCons(
                    node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1,
                    name=f"C_{u}_{v}"
                )

        for i, clique in enumerate(cliques):
            model.addCons(
                quicksum(node_vars[f"x{node}"] for node in clique) <= 1,
                name=f"Clique_{i}"
            )

        for u, v in G.edges:
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] <= 1 + piecewise_vars[f"z{u}_{v}"],
                name=f"PL1_{u}_{v}"
            )
            model.addCons(
                node_vars[f"x{u}"] + node_vars[f"x{v}"] >= 2 * piecewise_vars[f"z{u}_{v}"],
                name=f"PL2_{u}_{v}"
            )

        for u, v in G.edges:
            model.addCons(
                edge_vars[f"y{u}_{v}"] >= renewable_energy_vars[f"re_{u}_{v}"],
                name=f"RE_{u}_{v}"
            )

        for node in G.nodes:
            model.addCons(
                energy_vars[f"e{node}"] + demand_level[f"dem_{node}"] == 0,
                name=f"Node_{node}_energy"
            )

        for u, v in G.edges:
            model.addCons(
                G[u][v]['cost'] * terrain_difficulty[(u, v)] * weather_conditions[(u, v)] * edge_vars[f"y{u}_{v}"] <= yearly_budget,
                name=f"Budget_{u}_{v}"
            )

        model.addCons(
            yearly_budget <= self.yearly_budget_limit,
            name="Yearly_budget_limit"
        )

        # Add new nutrient constraints from Meal Planning
        for n in range(n_nutrients):
            model.addCons(
                quicksum(meals[f"m_{m}"][n] * meal_vars[f"m_{m}"] for m in range(n_meals)) >= required_nutrients[n],
                name=f"nutrient_{n}_requirement"
            )

        # Add meal diversity constraint
        model.addCons(
            quicksum(meal_vars[f"m_{m}"] for m in range(n_meals)) <= self.max_meals_per_day,
            name="meal_selection_limit"
        )

        model.setObjective(objective_expr, "maximize")
        
        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time

if __name__ == '__main__':
    seed = 42
    parameters = {
        'min_n': 25,
        'max_n': 731,
        'er_prob': 0.24,
        'set_type': 'SET1',
        'set_param': 2025.0,
        'alpha': 0.52,
        'renewable_percentage': 0.59,
        'carbon_limit': 750,
        'yearly_budget_limit': 5000,
        'min_meals': 200,
        'max_meals': 1050,
        'min_nutrients': 12,
        'max_nutrients': 75,
        'max_meals_per_day': 120,
    }
    gisp = GISP(parameters, seed=seed)
    instance = gisp.generate_instance()
    solve_status, solve_time = gisp.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")