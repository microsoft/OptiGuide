import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class HealthcareFacilityLocation:
    def __init__(self, parameters, seed=None):
        for key, value in parameters.items():
            setattr(self, key, value)
        self.seed = seed
        if self.seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_instance(self):
        assert self.totalSites > 0 and self.totalNodes > self.totalSites
        assert self.maxTravelDistances > 0 and self.facilityBudgetCost >= 0

        setup_costs = np.random.randint(10000, 50000, self.totalSites)
        on_site_costs = np.random.uniform(500, 2000, self.totalSites)
        node_demands = np.random.randint(50, 500, self.totalNodes)
        team_assignment_costs = np.random.uniform(100, 500, (self.totalSites, self.totalNodes))
        
        travel_distances = np.random.uniform(1, self.maxTravelDistances, (self.totalSites, self.totalNodes))
        
        shelf_life = np.random.randint(5, 30, (self.totalSites, self.totalNodes))
        holding_costs = np.random.uniform(50, 150, self.totalSites)
        capacities = np.random.randint(100, 500, self.totalSites)

        self.travel_graph = self.generate_random_graph()
        facility_budgets = np.random.randint(1000, 10000, self.totalSites)
        
        return {
            "setup_costs": setup_costs,
            "on_site_costs": on_site_costs,
            "node_demands": node_demands,
            "team_assignment_costs": team_assignment_costs,
            "travel_distances": travel_distances,
            "facility_budgets": facility_budgets,
            "shelf_life": shelf_life,
            "holding_costs": holding_costs,
            "capacities": capacities,
            "graph": self.travel_graph,
        }
    
    def generate_random_graph(self):
        n_nodes = self.totalNodes
        G = nx.barabasi_albert_graph(n=n_nodes, m=3, seed=self.seed)
        return G

    def solve(self, instance):
        setup_costs = instance['setup_costs']
        on_site_costs = instance['on_site_costs']
        node_demands = instance['node_demands']
        team_assignment_costs = instance['team_assignment_costs']
        travel_distances = instance['travel_distances']
        facility_budgets = instance['facility_budgets']
        shelf_life = instance['shelf_life']
        holding_costs = instance['holding_costs']
        capacities = instance['capacities']
        G = instance['graph']

        model = Model("HealthcareFacilityLocation")

        totalSites = len(setup_costs)
        totalNodes = len(node_demands)

        # Variables
        facility_vars = {s: model.addVar(vtype="B", name=f"Facility_{s}") for s in range(totalSites)}
        node_coverage_vars = {(s, n): model.addVar(vtype="B", name=f"NodeCoverage_{s}_{n}") for s in range(totalSites) for n in range(totalNodes)}
        team_assignment_vars = {(s, n): model.addVar(vtype="B", name=f"TeamAssignment_{s}_{n}") for s in range(totalSites) for n in range(totalNodes)}
        shelf_life_vars = {(s, n): model.addVar(vtype="C", name=f"ShelfLife_{s}_{n}", lb=0) for s in range(totalSites) for n in range(totalNodes)}
        delivery_time_vars = {(s, n): model.addVar(vtype="C", name=f"DeliveryTime_{s}_{n}", lb=0) for s in range(totalSites) for n in range(totalNodes)}
        inventory_vars = {s: model.addVar(vtype="C", name=f"Inventory_{s}", lb=0) for s in range(totalSites)}

        cost_vars = {s: model.addVar(vtype="C", name=f"SetupCost_{s}", lb=0) for s in range(totalSites)}
        
        required_teams = {n: model.addVar(vtype="B", name=f"RequiredTeams_{n}") for n in range(totalNodes)}

        # Constraints
        
        # Travel distance limits
        for s in range(totalSites):
            model.addCons(
                quicksum(travel_distances[s, n] * node_coverage_vars[s, n] for n in range(totalNodes)) <= self.maxTravelDistances,
                f"TravelLimit_{s}"
            )

        # Node coverage
        for n in range(totalNodes):
            model.addCons(
                quicksum(node_coverage_vars[s, n] for s in range(totalSites)) >= 1,
                f"NodeCoverage_{n}"
            )

        # Budget constraints
        for s in range(totalSites):
            model.addCons(
                sum(node_coverage_vars[s, n] for n in range(totalNodes)) * on_site_costs[s] <= facility_budgets[s],
                f"Budget_{s}"
            )

        # Linking facility variable with node coverage
        for s in range(totalSites):
            for n in range(totalNodes):
                model.addCons(
                    node_coverage_vars[s, n] <= facility_vars[s],
                    f"Coverage_{s}_{n}"
                )

        # Team assignment to cover required teams
        for n in range(totalNodes):
            model.addCons(
                quicksum(team_assignment_vars[s, n] for s in range(totalSites)) >= required_teams[n],
                f"TeamRequirement_{n}"
            )

        # Shelf life constraints
        for s in range(totalSites):
            for n in range(totalNodes):
                model.addCons(
                    shelf_life_vars[s, n] <= shelf_life[s, n] - delivery_time_vars[s, n],
                    f"ShelfLifeConstraint_{s}_{n}"
                )

        # Delivery time constraints
        for s in range(totalSites):
            for n in range(totalNodes):
                model.addCons(
                    delivery_time_vars[s, n] <= self.maxTravelDistances,
                    f"DeliveryTimeConstraint_{s}_{n}"
                )

        # Capacity constraints
        for s in range(totalSites):
            model.addCons(
                inventory_vars[s] <= capacities[s],
                f"Capacity_{s}"
            )

        # Objective to minimize cost and max shelf life
        model.setObjective(
            quicksum(setup_costs[s] * facility_vars[s] for s in range(totalSites)) +
            quicksum(team_assignment_costs[s, n] * team_assignment_vars[s, n] for s in range(totalSites) for n in range(totalNodes)) +
            quicksum(holding_costs[s] * inventory_vars[s] for s in range(totalSites)),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'totalSites': 30,
        'totalTeams': 2,
        'totalNodes': 135,
        'facilityBudgetCost': 200000,
        'maxTravelDistances': 250,
    }

    facility_solver = HealthcareFacilityLocation(parameters, seed=42)
    instance = facility_solver.generate_instance()
    solve_status, solve_time, objective_value = facility_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")