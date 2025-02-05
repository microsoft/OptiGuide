import random
import time
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

class SimplifiedHealthcareFacilityLocation:
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
        
        travel_distances = np.abs(np.random.normal(loc=400, scale=150, size=(self.totalSites, self.totalNodes)))
        travel_distances = np.where(travel_distances > self.maxTravelDistances, self.maxTravelDistances, travel_distances)
        travel_distances = np.where(travel_distances < 1, 1, travel_distances)

        self.travel_graph = self.generate_random_graph()
        
        return {
            "setup_costs": setup_costs,
            "on_site_costs": on_site_costs,
            "node_demands": node_demands,
            "travel_distances": travel_distances,
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
        travel_distances = instance['travel_distances']
        G = instance['graph']

        model = Model("SimplifiedHealthcareFacilityLocation")

        totalSites = len(setup_costs)
        totalNodes = len(node_demands)

        facility_vars = {s: model.addVar(vtype="B", name=f"Facility_{s}") for s in range(totalSites)}
        node_coverage_vars = {(s, n): model.addVar(vtype="B", name=f"NodeCoverage_{s}_{n}") for s in range(totalSites) for n in range(totalNodes)}

        for s in range(totalSites):
            model.addCons(
                quicksum(travel_distances[s, n] * node_coverage_vars[s, n] for n in range(totalNodes)) <= self.maxTravelDistances,
                f"TravelLimit_{s}"
            )

        for n in range(totalNodes):
            model.addCons(
                quicksum(node_coverage_vars[s, n] for s in range(totalSites)) >= 1,
                f"NodeCoverage_{n}"
            )

        for s in range(totalSites):
            for n in range(totalNodes):
                model.addCons(
                    node_coverage_vars[s, n] <= facility_vars[s],
                    f"Coverage_{s}_{n}"
                )

        # Symmetry Breaking Constraints
        for s in range(totalSites - 1):
            model.addCons(
                facility_vars[s] >= facility_vars[s + 1],
                f"Symmetry_{s}"
            )
        
        # Hierarchical Demand Fulfillment
        for n in range(totalNodes):
            for s1 in range(totalSites):
                for s2 in range(s1 + 1, totalSites):
                    model.addCons(
                        node_coverage_vars[s2, n] <= node_coverage_vars[s1, n] + int(travel_distances[s1, n] <= self.maxTravelDistances),
                        f"Hierarchical_{s1}_{s2}_{n}"
                    )

        model.setObjective(
            quicksum(setup_costs[s] * facility_vars[s] for s in range(totalSites)) +
            quicksum(travel_distances[s, n] * node_coverage_vars[s, n] for s in range(totalSites) for n in range(totalNodes)),
            "minimize"
        )

        start_time = time.time()
        model.optimize()
        end_time = time.time()
        
        return model.getStatus(), end_time - start_time, model.getObjVal()


if __name__ == '__main__':
    seed = 42
    parameters = {
        'totalSites': 15,
        'totalNodes': 70,
        'facilityBudgetCost': 150000,
        'maxTravelDistances': 800,
    }

    simplified_facility_solver = SimplifiedHealthcareFacilityLocation(parameters, seed=42)
    instance = simplified_facility_solver.generate_instance()
    solve_status, solve_time, objective_value = simplified_facility_solver.solve(instance)

    print(f"Solve Status: {solve_status}")
    print(f"Solve Time: {solve_time:.2f} seconds")
    print(f"Objective Value: {objective_value:.2f}")